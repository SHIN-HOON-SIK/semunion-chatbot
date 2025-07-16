# -*- coding: utf-8 -*-

import os
import sys
import re
import hashlib
from pathlib import Path

import streamlit as st
from PyPDF2 import PdfReader
from pptx import Presentation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --------------------------------------------------------------------------
# [1. 기본 유틸리티 함수 정의]
# --------------------------------------------------------------------------

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def safe_unicode(text: str) -> str:
    """안전하게 유니코드로 변환하는 함수"""
    if not isinstance(text, str):
        text = str(text)
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

def clean_text(text):
    """텍스트에서 제어 문자 등 불필요한 문자를 제거하는 함수"""
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", text)
    text = re.sub(r"[\ud800-\udfff]", "", text)
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore").strip()

def extract_text_from_pdf(path: Path) -> str:
    """PDF 파일에서 텍스트를 추출하는 함수"""
    try:
        reader = PdfReader(str(path))
        pages = [clean_text(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        st.warning(f"[PDF 추출 실패] {path.name}: {e}")
        return ""

def extract_text_from_pptx(path: Path) -> str:
    """PPTX 파일에서 텍스트를 추출하는 함수"""
    try:
        prs = Presentation(str(path))
        slides = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slides.append(clean_text(shape.text))
        return "\n".join(slides)
    except Exception as e:
        st.warning(f"[PPTX 추출 실패] {path.name}: {e}")
        return ""

def compute_file_hash(file_paths):
    """파일 목록의 해시를 계산하여 파일 변경을 감지하는 함수"""
    hash_md5 = hashlib.md5()
    for path in sorted(file_paths):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()


# --------------------------------------------------------------------------
# [2. LangChain 및 AI 모델 관련 기능 정의]
# --------------------------------------------------------------------------

@st.cache_resource
def load_all_documents_with_hash(file_paths, file_hash):
    """지정된 경로의 모든 문서를 불러와 LangChain Document 객체로 변환"""
    documents = []
    for path in file_paths:
        text = ""
        if path.suffix == ".pdf":
            text = extract_text_from_pdf(path)
        elif path.suffix == ".pptx":
            text = extract_text_from_pptx(path)
        
        if text.strip():
            doc = Document(page_content=text, metadata={"source": str(path.name)})
            documents.append(doc)
        else:
            st.warning(f"[삭제] {path.name} 의 텍스트가 비어 있습니다.")
    return documents

@st.cache_resource
def split_documents_into_chunks(_documents):
    """문서를 적절한 크기의 청크로 분할"""
    total_length = sum(len(doc.page_content) for doc in _documents)
    avg_length = total_length // len(_documents) if _documents else 0
    chunk_size, overlap = (1500, 300) if avg_length > 6000 else (1000, 200) if avg_length > 3000 else (700, 200)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(_documents)

@st.cache_resource
def create_vector_store(_chunks, _embedding_model):
    """분할된 청크로부터 FAISS 벡터 데이터베이스를 생성"""
    try:
        return FAISS.from_documents(_chunks, _embedding_model)
    except Exception as e:
        st.error(f"❌ FAISS 벡터 DB 생성 중 오류 발생: {safe_unicode(str(e))}")
        st.stop()

# LLM에게 역할을 부여하는 시스템 프롬프트
QA_SYSTEM_PROMPT = """
너는 반드시 PDF/PPTX 문서에 포함된 내용만 바탕으로 답해야 해.
문서에 명시적 언급이 없거나 애매한 경우 '문서에 해당 정보가 없습니다.'라고 답해.
"""

# 질문과 컨텍스트를 결합하는 프롬프트 템플릿
QA_QUESTION_PROMPT = PromptTemplate(
    template=QA_SYSTEM_PROMPT + "\n\n{context}\n\n질문: {question}\n답변:",
    input_variables=["context", "question"]
)

@st.cache_resource
def initialize_qa_chain(all_paths, api_key):
    """모든 구성요소를 초기화하여 QA 체인을 생성"""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o", temperature=0)
    file_hash = compute_file_hash(all_paths)
    docs = load_all_documents_with_hash(all_paths, file_hash)
    if not docs:
        st.error("문서를 불러오지 못했습니다.")
        st.stop()
    
    chunks = split_documents_into_chunks(docs)

    # 하이브리드 검색 설정 (BM25 + FAISS)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 20

    faiss_vectorstore = create_vector_store(chunks, embeddings)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 20})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    # --- 다중 질문 생성 기능 (안정적인 기본 버전 사용) ---
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever, llm=llm
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=multi_query_retriever,
        chain_type_kwargs={"prompt": QA_QUESTION_PROMPT},
        return_source_documents=True
    )

# --------------------------------------------------------------------------
# [3. Streamlit 앱 실행 부분]
# --------------------------------------------------------------------------

# [API 키 설정]
openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not openai_api_key:
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error("❌ OpenAI API 키가 설정되지 않았습니다. Streamlit Secrets에 키를 등록해주세요.")
        st.stop()

# [페이지 설정 및 제목]
st.set_page_config(page_title="삼성전기 종중노조 상담사", layout="centered", page_icon="logo_union_hands.png")
st.markdown("""
<div style='display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 10px;'>
    <img src="https://raw.githubusercontent.com/SHIN-HOON-SIK/semunion-chatbot/main/logo_union_hands.png" width="50"/>
    <h1 style='color: #0d1a44; margin: 0;'>삼성전기 종중노조 상담사</h1>
</div>
""", unsafe_allow_html=True)
st.write("노조 집행부에서 등록한 PDF 및 PPTX 문서 기반으로 질문하신 내용에 답변해 드립니다.")

# [문서 경로 설정]
base_dir = Path(__file__).parent
data_dir = base_dir / "data"
doc_files = ["policy_agenda_250627.pdf", "union_meeting_250704.pdf", "SEMUNION_DATA_BASE.pdf", "SEMUNION_DATA_BASE.pptx"]
doc_paths = [data_dir / name for name in doc_files if (data_dir / name).exists()]

# [QA 체인 초기화]
try:
    qa_chain = initialize_qa_chain(all_paths=doc_paths, api_key=openai_api_key)
except Exception as e:
    st.error(f"⚠️ 앱 초기화 중 오류가 발생했습니다: {e}")
    st.stop()

# [사용자 질문 입력 및 답변 처리]
user_query = st.text_input("궁금하신 내용은 아래 창에 질문을 해보세요!", placeholder="여기에 최대한 구체적으로 질문 부탁드립니다.")

if user_query.strip():
    with st.spinner("답변 생성 중..."):
        try:
            result = qa_chain.invoke({"query": user_query})
            answer = result["result"]
            
            if not answer or "문서에 해당 정보가 없습니다" in answer:
                st.info("죄송하지만 업로드된 문서 내에서 관련된 내용을 찾을 수 없습니다. 또는 조금 더 구체적으로 질문 부탁드립니다.")
            else:
                st.success(answer)

            with st.expander("📄 답변 근거 문서"):
                source_docs = result.get("source_documents", [])
                if source_docs:
                    for i, doc in enumerate(source_docs):
                        name = Path(doc.metadata.get("source", "알 수 없는 파일")).name
                        st.markdown(f"**문서 {i+1}:** `{name}`")
                        preview = doc.page_content[:500] + "..."
                        st.text(preview)
                else:
                    st.write("답변에 대한 근거 문서를 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"❌ 답변 생성 중 오류가 발생했습니다: {e}")
