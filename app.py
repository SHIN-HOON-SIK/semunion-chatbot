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

# ✅ 3. PDF 전처리용 문자열 클리너
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
            doc = Document(page_content=text, metadata={"source": str(path.name)})
            documents.append(doc)
        else:
            st.warning(f"[생략] {path.name} 의 텍스트가 비어 있습니다.")
    return documents

# ✅ 7. chunk 분리
@st.cache_resource
def split_documents_into_chunks(documents):
    total_length = sum(len(doc.page_content) for doc in documents)
    avg_length = total_length // len(documents) if documents else 0

    if avg_length > 6000:
        chunk_size, overlap = 1500, 300
    elif avg_length > 3000:
        chunk_size, overlap = 1000, 200
    else:
        chunk_size, overlap = 700, 100

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(documents)

# ✅ 8. FAISS 벡터 DB
@st.cache_resource
def create_vector_store(chunks, embedding_model):
    try:
        for doc in chunks:
            doc.page_content = safe_unicode(doc.page_content)
        return FAISS.from_documents(chunks, embedding_model)
    except Exception as e:
        st.error(f"❌ FAISS 벡터 DB 생성 중 오류 발생: {e}")
        st.stop()

# ✅ 9. QA 체인
@st.cache_resource
def initialize_qa_chain(pdf_paths):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    file_hash = compute_file_hash(pdf_paths)
    docs = load_all_documents_with_hash(pdf_paths, file_hash)
    if not docs:
        st.error("PDF 문서를 불러올 수 없습니다.")
        st.stop()
    chunks = split_documents_into_chunks(docs)
    db = create_vector_store(chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# ✅ 10. 질문 확장
@st.cache_resource
def get_query_expander():
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    def expand(query: str) -> str:
        try:
            prompt = HumanMessage(content=safe_unicode(
                "사용자의 질문을 PDF 내용과 잘 매칭되도록 구체적이고 명확하게 바꿔줘. "
                "PDF 용어와 표현을 반영하고, 필요한 부연도 추가해도 좋아. "
                f"질문: {query}"
            ))
            response = llm.invoke([prompt])
            return safe_unicode(response.content.strip())
        except Exception as e:
            st.warning(f"❕ 질문 확장 실패: {e!r}")
            return query
    return expand
