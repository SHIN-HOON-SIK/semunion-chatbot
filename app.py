# union_chatbot_app.py
# -*- coding: utf-8 -*-

import os
import sys
import re
import hashlib
from pathlib import Path
import streamlit as st
from PyPDF2 import PdfReader
from pptx import Presentation  # ✅ 이 라인을 사용하기 위해 python-pptx 설치 필요
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, HumanMessage
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def safe_unicode(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", text)
    text = re.sub(r"[\ud800-\udfff]", "", text)
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore").strip()

def extract_text_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = [clean_text(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        st.warning(f"[PDF 추출 실패] {path.name}: {e}")
        return ""

def extract_text_from_pptx(path: Path) -> str:
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
    hash_md5 = hashlib.md5()
    for path in sorted(file_paths):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()

@st.cache_resource
def load_all_documents_with_hash(file_paths, file_hash):
    documents = []
    for path in file_paths:
        if path.suffix == ".pdf":
            text = extract_text_from_pdf(path)
        elif path.suffix == ".pptx":
            text = extract_text_from_pptx(path)
        else:
            continue
        if text.strip():
            doc = Document(page_content=safe_unicode(text), metadata={"source": str(path.name)})
            documents.append(doc)
        else:
            st.warning(f"[삭락] {path.name} 의 텍스트가 비어 있습니다.")
    return documents

@st.cache_resource
def split_documents_into_chunks(_documents):
    total_length = sum(len(doc.page_content) for doc in _documents)
    avg_length = total_length // len(_documents) if _documents else 0
    chunk_size, overlap = (1500, 300) if avg_length > 6000 else (1000, 200) if avg_length > 3000 else (700, 200)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(_documents)

@st.cache_resource
def create_vector_store(_chunks, _embedding_model):
    try:
        return FAISS.from_documents(
            [Document(page_content=safe_unicode(doc.page_content), metadata=doc.metadata) for doc in _chunks],
            _embedding_model
        )
    except Exception as e:
        st.error(f"❌ FAISS 벡터 DB 생성 중 오류 발생: {safe_unicode(str(e))}")
        st.stop()

QA_SYSTEM_PROMPT = """
너는 반드시 PDF/PPTX 문서에 포함된 내용만 바탕으로 답해야 해.
문서에 명시적 언급이 없거나 애매한 경우 '문서에 해당 정보가 없습니다.'라고 답해.
"""

QA_QUESTION_PROMPT = PromptTemplate(
    template=QA_SYSTEM_PROMPT + "\n\n{context}\n\n질문: {question}\n답변:",
    input_variables=["context", "question"]
)

@st.cache_resource
def initialize_qa_chain(all_paths):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    file_hash = compute_file_hash(all_paths)
    docs = load_all_documents_with_hash(all_paths, file_hash)
    if not docs:
        st.error("문서를 불러오지 못했습니다.")
        st.stop()
    chunks = split_documents_into_chunks(docs)
    db = create_vector_store(chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_QUESTION_PROMPT},
        return_source_documents=True
    )

@st.cache_resource
def get_query_expander():
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    def expand(query: str) -> str:
        try:
            prompt = HumanMessage(content=safe_unicode(
                "다음 단어나 문장을 PDF/PPTX 검색에 최적화되도록 바꿔줘. "
                "문서에서 자주 등장하는 표현을 반영해서 재작성해줘. 동의어를 쓰지 말고 문서 언어 그대로 사용해. "
                f"질문: {query}"
            ))
            response = llm.invoke([prompt])
            return safe_unicode(response.content.strip())
        except Exception as e:
            st.warning(f"❕ 질문 확장 실패: {safe_unicode(str(e))}")
            return query
    return expand

openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not openai_api_key:
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

st.set_page_config(page_title="삼성전기 종중노조 상담사", layout="centered", page_icon="logo_union_hands.png")
st.markdown("""
<div style='display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 10px;'>
    <img src="https://raw.githubusercontent.com/SHIN-HOON-SIK/semunion-chatbot/main/logo_union_hands.png" width="50"/>
    <h1 style='color: #0d1a44; margin: 0;'>삼성전기 종중노조 상담사</h1>
</div>
""", unsafe_allow_html=True)

st.write("PDF 및 PPTX 문서 기반 질문에 대해 GPT가 답변해 드립니다.")

base_dir = Path(__file__).parent
data_dir = base_dir / "data"
doc_files = ["policy_agenda_250627.pdf", "union_meeting_250704.pdf", "SEMUNION_DATA_BASE.pdf", "SEMUNION_DATA_BASE.pptx"]
doc_paths = [data_dir / name for name in doc_files if (data_dir / name).exists()]

try:
    query_expander = get_query_expander()
    qa_chain = initialize_qa_chain(doc_paths)
except Exception as e:
    st.error(f"⚠️ 초기화 실패: {safe_unicode(str(e))}")
    st.stop()

user_query = st.text_input("무엇이 궁금하시나요?", placeholder="예: 집행부 구성은?")
if user_query.strip():
    query = query_expander(user_query)
    with st.spinner("답변 생성 중..."):
        try:
            result = qa_chain.invoke({"query": query})
            answer = safe_unicode(result["result"])
            if not answer or "문서에 해당 정보가 없습니다" in answer:
                st.info("죄송하지만 업로드된 문서 내에서 관련된 내용을 찾을 수 없습니다.")
            else:
                st.success(answer)

            with st.expander("📄 답변 근거 문서 보기"):
                for i, doc in enumerate(result["source_documents"]):
                    name = Path(doc.metadata.get("source", "알 수 없는 파일")).name
                    st.markdown(f"**문서 {i+1}:** `{name}`")
                    preview = safe_unicode(doc.page_content[:500]) + "..."
                    st.text(preview)
        except Exception as e:
            st.error(f"❌ 답변 생성 실패: {safe_unicode(str(e))}")
