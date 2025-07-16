# union_chatbot_app.py
# -*- coding: utf-8 -*-

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

# 🎯 1. Windows 인코딩 문제 해결
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ✅ 2. 안전한 유니코드 정리 함수

def safe_unicode(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

# ✅ 3. PDF 전체 문자열 클리너 함수

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", text)
    text = re.sub(r"[\ud800-\udfff]", "", text)
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore").strip()

# ✅ 4. PDF 텍스트 추출

def extract_text_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            raw = page.extract_text() or ""
            pages.append(clean_text(raw))
        return "\n".join(pages)
    except Exception as e:
        st.warning(f"[PDF 추출 실패] {path.name}: {e}")
        return ""

# ✅ 5. 파일 해시

def compute_file_hash(file_paths):
    hash_md5 = hashlib.md5()
    for path in sorted(file_paths):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()

# ✅ 6. 문서 로딩

@st.cache_resource
def load_all_documents_with_hash(pdf_paths, file_hash):
    documents = []
    for path in pdf_paths:
        text = extract_text_from_pdf(path)
        if text.strip():
            doc = Document(page_content=safe_unicode(text), metadata={"source": str(path.name)})
            documents.append(doc)
        else:
            st.warning(f"[삭락] {path.name} 의 텍스트가 비어 있습니다.")
    return documents

# ✅ 7. chunk 분리

@st.cache_resource
def split_documents_into_chunks(_documents):
    total_length = sum(len(doc.page_content) for doc in _documents)
    avg_length = total_length // len(_documents) if _documents else 0

    if avg_length > 6000:
        chunk_size, overlap = 1500, 300
    elif avg_length > 3000:
        chunk_size, overlap = 1000, 200
    else:
        chunk_size, overlap = 700, 100

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(_documents)

# ✅ 8. FAISS 베터 DB

@st.cache_resource
def create_vector_store(_chunks, _embedding_model):
    try:
        for doc in _chunks:
            doc.page_content = safe_unicode(doc.page_content)
        return FAISS.from_documents(
            [Document(page_content=safe_unicode(doc.page_content), metadata=doc.metadata) for doc in _chunks],
            _embedding_model
        )
    except Exception as e:
        st.error(f"❌ FAISS 베터 DB 생성 중 오류 발생: {safe_unicode(str(e))}")
        st.stop()

# ✅ 9. QA 체인

@st.cache_resource
def initialize_qa_chain(pdf_paths):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    file_hash = compute_file_hash(pdf_paths)
    docs = load_all_documents_with_hash(pdf_paths, file_hash)
    if not docs:
        st.error("PDF 문서를 불러오지 못했습니다.")
        st.stop()
    chunks = split_documents_into_chunks(docs)
    db = create_vector_store(chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# ✅ 10. 질문 확장

@st.cache_resource
def get_query_expander():
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    def expand(query: str) -> str:
        try:
            prompt = HumanMessage(content=safe_unicode(
                "사용자의 질문을 PDF 내용과 잘 매칭되도록 구체적이고 명확한 문장으로 바꾼 해주어."
                f" 질문: {query}"
            ))
            response = llm.invoke([prompt])
            return safe_unicode(response.content.strip())
        except Exception as e:
            st.warning(f"❕ 질문 확장 실패: {safe_unicode(str(e))}")
            return query
    return expand

# ✅ 11. OpenAI 키 설정

openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not openai_api_key:
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

# ✅ 12. Streamlit UI

st.set_page_config(page_title="삼성전기 종중노조 상담사", layout="centered", page_icon="logo_union_hands.png")
st.markdown("""
<div style='display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 10px;'>
    <img src="https://raw.githubusercontent.com/SHIN-HOON-SIK/semunion-chatbot/main/logo_union_hands.png" width="50"/>
    <h1 style='color: #0d1a44; margin: 0;'>삼성전기 종중노조 상담사</h1>
</div>
""", unsafe_allow_html=True)

st.write("PDF 문서 기반 질문에 대해 GPT가 답변해 드림니다.")

base_dir = Path(__file__).parent
pdf_dir = base_dir / "data"
pdf_files = ["policy_agenda_250627.pdf", "union_meeting_250704.pdf", "SEMUNION_DATA_BASE.pdf"]
pdf_paths = [pdf_dir / name for name in pdf_files]

try:
    query_expander = get_query_expander()
    qa_chain = initialize_qa_chain(pdf_paths)
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
            st.success(answer or "정보를 찾을 수 없습니다.")

            with st.expander("파본 구글 문서 보기"):
                for i, doc in enumerate(result["source_documents"]):
                    name = Path(doc.metadata.get("source", "알 수 없는 파일")).name
                    st.markdown(f"**문서 {i+1}:** `{name}`")
                    preview = safe_unicode(doc.page_content[:500]) + "..."
                    st.text(preview)
        except Exception as e:
            st.error(f"❌ 답변 생성 실패: {safe_unicode(str(e))}")
