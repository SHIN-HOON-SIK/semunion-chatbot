# union_chatbot_app.py
# -*- coding: utf-8 -*-

import os
import re
import hashlib
import sys
from pathlib import Path
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, Document

# ✅ 환경: Windows 인코딩 안전 처리
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ✅ 유니코드 정제 함수
def safe_unicode(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

# ✅ PDF 텍스트 정제 함수
def clean_text(text):
    text = text.replace("\x00", "")
    text = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", text)  # 제어문자 제거
    text = re.sub(r"[\ud800-\udfff]", "", text)  # surrogate 문자 제거
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore").strip()

# ✅ PDF 텍스트 추출
def extract_text_from_pdf(path):
    try:
        reader = PdfReader(str(path))
        pages = [clean_text(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        st.warning(f"[PDF 추출 오류] {path.name}: {e}")
        return ""

# ✅ 파일 변경 감지를 위한 해시
def compute_file_hash(file_paths):
    h = hashlib.md5()
    for path in sorted(file_paths):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
    return h.hexdigest()

# ✅ 문서 로딩
@st.cache_resource
def load_all_documents_with_hash(pdf_paths, file_hash):
    docs = []
    for path in pdf_paths:
        text = extract_text_from_pdf(path)
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": str(path.name)}))
        else:
            st.warning(f"⚠️ {path.name} 텍스트 없음 → 생략")
    return docs

# ✅ 텍스트 분할
@st.cache_resource
def split_documents_into_chunks(_documents):
    total_length = sum(len(doc.page_content) for doc in _documents)
    avg = total_length // len(_documents) if _documents else 0
    if avg > 6000:
        chunk_size, overlap = 1500, 300
    elif avg > 3000:
        chunk_size, overlap = 1000, 200
    else:
        chunk_size, overlap = 700, 100
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(_documents)

# ✅ 벡터 DB 생성
@st.cache_resource
def create_vector_store(_chunks, _embedding_model):
    try:
        for doc in _chunks:
            doc.page_content = safe_unicode(doc.page_content)
        return FAISS.from_documents(_chunks, _embedding_model)
    except Exception as e:
        st.error(f"❌ 벡터 DB 생성 오류: {safe_unicode(str(e))}")
        st.stop()

# ✅ QA 체인 초기화
@st.cache_resource
def initialize_qa_chain():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    full_paths = [PDF_FILES_DIR / name for name in PDF_FILES]
    file_hash = compute_file_hash(full_paths)
    docs = load_all_documents_with_hash(full_paths, file_hash)
    if not docs:
        st.error("❌ 유효한 문서를 찾을 수 없습니다.")
        st.stop()
    chunks = split_documents_into_chunks(docs)
    db = create_vector_store(chunks, embeddings)
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 6}), return_source_documents=True)

# ✅ 질문 확장 GPT
@st.cache_resource
def get_query_expander():
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    def expand(query):
        try:
            prompt = HumanMessage(content=safe_unicode(
                "사용자의 질문을 PDF 내용과 매칭되도록 명확하게 다시 작성해줘.\n"
                f"질문: {query}"
            ))
            res = llm.invoke([prompt])
            return safe_unicode(res.content.strip())
        except Exception as e:
            st.warning(f"❕ 질문 확장 오류: {safe_unicode(str(e))}")
            return query
    return expand

# ✅ OpenAI API 키 로딩
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"].strip()
except Exception:
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not openai_api_key or not openai_api_key.startswith("sk-"):
    st.error("❌ OpenAI API 키가 유효하지 않거나 누락됨 (sk-로 시작해야 함)")
    st.stop()

# ✅ 설정
BASE_DIR = Path(__file__).parent
PDF_FILES_DIR = BASE_DIR / "data"
PDF_FILES = [
    "policy_agenda_250627.pdf",
    "union_meeting_250704.pdf",
    "SEMUNION_DATA_BASE.pdf"
]

# ✅ UI 구성
st.set_page_config(page_title="삼성전기 존중노조 상담사", layout="centered", page_icon="🤖")
st.title("🤖 삼성전기 존중노조 상담사")
st.write("PDF 문서 기반으로 노조 및 회사 관련 질문에 답변해 드립니다.")

# ✅ 체인 초기화
try:
    query_expander = get_query_expander()
    qa_chain = initialize_qa_chain()
except Exception as e:
    st.error(f"⚠️ 초기화 실패: {safe_unicode(str(e))}")
    st.stop()

# ✅ 사용자 쿼리 처리
raw_query = st.text_input("무엇이 궁금하신가요?", placeholder="예: 신훈식 위원장에 대해 알려줘")
query = query_expander(raw_query.strip()) if raw_query.strip() else ""

if query:
    with st.spinner("🤖 답변 생성 중..."):
        try:
            result = qa_chain.invoke({"query": query})
            answer = safe_unicode(result["result"])
            if not answer or "없" in answer:
                st.info("🛑 문서 내 해당 정보가 없습니다. 집행부에 업데이트 요청 가능합니다.")
            else:
                st.success(answer)
            
            with st.expander("📎 답변 근거 문서 보기"):
                for i, doc in enumerate(result["source_documents"]):
                    name = Path(doc.metadata.get("source", "알 수 없음")).name
                    preview = safe_unicode(doc.page_content.strip())[:400] + "..."
                    st.markdown(f"**문서 {i+1}:** `{name}`")
                    st.text(preview)
                    st.markdown("---")
        except Exception as e:
            st.error(f"❌ 답변 생성 오류: {safe_unicode(str(e))}")

