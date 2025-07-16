# app.py - 전체 리팩터링 통합 버전

import os
import sys
import re
import hashlib
from pathlib import Path
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, HumanMessage
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ✅ 유니코드 및 문자열 처리

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
    return safe_unicode(text.strip())

# ✅ PDF 텍스트 및 메타데이터 추출

def extract_text_from_pdf(path: Path) -> list:
    try:
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            pages.append(Document(
                page_content=clean_text(raw),
                metadata={"source": path.name, "page_number": i + 1}
            ))
        return pages
    except Exception as e:
        st.warning(f"[PDF 추출 실패] {path.name}: {e}")
        return []

# ✅ 해시 기반 중복 방지 캐싱

def compute_file_hash(file_paths):
    hash_md5 = hashlib.md5()
    for path in sorted(file_paths):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()

@st.cache_resource
def load_documents_with_metadata(pdf_paths):
    documents = []
    for path in pdf_paths:
        pages = extract_text_from_pdf(path)
        documents.extend(pages)
    return documents

# ✅ chunk 크기 조정 (자동 + 수동)

def get_chunk_config(documents):
    total_length = sum(len(doc.page_content) for doc in documents)
    avg_length = total_length // len(documents) if documents else 0
    if avg_length > 6000:
        return 1500, 300
    elif avg_length > 3000:
        return 1000, 200
    else:
        return 700, 200

@st.cache_resource
def split_documents_into_chunks(documents, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(documents)

# ✅ 벡터 DB 생성

def initialize_vector_store(chunks, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    for doc in chunks:
        doc.page_content = safe_unicode(doc.page_content)
    return FAISS.from_documents(chunks, embeddings)

# ✅ 질문 확장

def get_query_expander(api_key):
    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o", temperature=0)
    def expand(query: str) -> str:
        try:
            prompt = HumanMessage(content=safe_unicode(
                "다음 문장을 PDF 검색에 최적화되도록 바꿔줘. PDF 내 자주 나오는 표현이나 문장 형태로 재작성하고, 문맥이 애매하면 부연 설명도 넣어줘.\n"
                f"질문: {query}"
            ))
            response = llm.invoke([prompt])
            return safe_unicode(response.content.strip())
        except Exception as e:
            st.warning(f"❕ 질문 확장 실패: {safe_unicode(str(e))}")
            return query
    return expand

# ✅ QA 체인 생성

def initialize_qa_chain(vector_store, api_key):
    system_prompt = """
너는 반드시 PDF 문서에 포함된 내용만 바탕으로 답해야 해.
문서에 명시적 언급이 없거나 애매한 경우 '문서에 해당 정보가 없습니다.'라고 답해.
"""
    qa_prompt = PromptTemplate(
        template=system_prompt + "\n\n{context}\n\n질문: {question}\n답변:",
        input_variables=["context", "question"]
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 15})
    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )

# ✅ Streamlit UI 구성
st.set_page_config(page_title="삼성전기 종중노조 상담사", layout="centered", page_icon="logo_union_hands.png")
st.markdown("""
<div style='display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 10px;'>
    <img src="https://raw.githubusercontent.com/SHIN-HOON-SIK/semunion-chatbot/main/logo_union_hands.png" width="50"/>
    <h1 style='color: #0d1a44; margin: 0;'>삼성전기 종중노조 상담사</h1>
</div>
""", unsafe_allow_html=True)

st.write("PDF 문서 기반 질문에 대해 GPT가 답변해 드립니다.")

openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not openai_api_key:
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

base_dir = Path(__file__).parent
pdf_dir = base_dir / "data"
pdf_files = ["policy_agenda_250627.pdf", "union_meeting_250704.pdf", "SEMUNION_DATA_BASE.pdf"]
pdf_paths = [pdf_dir / name for name in pdf_files]

try:
    documents = load_documents_with_metadata(pdf_paths)
    chunk_size, overlap = get_chunk_config(documents)
    chunk_size = st.sidebar.slider("Chunk 크기", 300, 2000, chunk_size, step=100)
    overlap = st.sidebar.slider("Overlap", 0, 500, overlap, step=50)
    chunks = split_documents_into_chunks(documents, chunk_size, overlap)
    vector_store = initialize_vector_store(chunks, openai_api_key)
    query_expander = get_query_expander(openai_api_key)
    qa_chain = initialize_qa_chain(vector_store, openai_api_key)
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
                st.info("문서 내에서 관련된 내용을 찾을 수 없습니다.")
            else:
                st.success(answer)

            with st.expander("📄 답변 근거 문서 보기"):
                for i, doc in enumerate(result["source_documents"]):
                    name = Path(doc.metadata.get("source", "알 수 없는 파일")).name
                    page = doc.metadata.get("page_number", "?")
                    st.markdown(f"**문서 {i+1}:** `{name}` (p.{page})")
                    preview = safe_unicode(doc.page_content[:500]) + "..."
                    st.text(preview)
        except Exception as e:
            st.error(f"❌ 답변 생성 실패: {safe_unicode(str(e))}")
