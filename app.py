# union_chatbot_app.py
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
from langchain.schema import Document, HumanMessage
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ... (기존 safe_unicode, clean_text, extract_text_from_pdf, extract_text_from_pptx, compute_file_hash 함수는 그대로 사용) ...

@st.cache_resource
def load_all_documents_with_hash(file_paths, file_hash):
    # ... (기존과 동일) ...
    documents = []
    for path in file_paths:
        if path.suffix == ".pdf":
            text = extract_text_from_pdf(path)
        elif path.suffix == ".pptx":
            text = extract_text_from_pptx(path)
        else:
            continue
        if text.strip():
            # 여기서 한 번만 clean_text와 safe_unicode를 적용하는 것을 고려
            doc = Document(page_content=text, metadata={"source": str(path.name)})
            documents.append(doc)
        else:
            st.warning(f"[삭제] {path.name} 의 텍스트가 비어 있습니다.")
    return documents


@st.cache_resource
def split_documents_into_chunks(_documents):
    # ... (기존과 동일) ...
    total_length = sum(len(doc.page_content) for doc in _documents)
    avg_length = total_length // len(_documents) if _documents else 0
    chunk_size, overlap = (1500, 300) if avg_length > 6000 else (1000, 200) if avg_length > 3000 else (700, 200)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(_documents)


@st.cache_resource
def create_vector_store(_chunks, _embedding_model):
    # ... (기존과 동일) ...
    try:
        return FAISS.from_documents(_chunks, _embedding_model)
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

    # ✅ BM25 + FAISS 하이브리드 검색 (EnsembleRetriever 사용)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    faiss_vectorstore = create_vector_store(chunks, embeddings)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 15})
    
    # EnsembleRetriever를 사용하여 두 리트리버 결합
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]  # BM25(키워드)와 FAISS(의미) 검색 결과의 가중치 조절
    )

    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ensemble_retriever, # 수정: ensemble_retriever 사용
        chain_type_kwargs={"prompt": QA_QUESTION_PROMPT},
        return_source_documents=True
    )

@st.cache_resource
def get_query_expander():
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    def expand(query: str) -> str:
        # 단어가 3개 이하인 짧은 질문에 대해서만 확장 시도
        if len(query.split()) <= 3:
            try:
                prompt_text = f"""
                '{query}'이라는 키워드가 포함된, 문서에서 실제 사용될 법한 자연스러운 질문 문장으로 만들어줘.
                예를 들어 '집행부'라는 키워드가 들어오면 '집행부 구성원은 누구인가요?' 와 같이 생성해줘.
                반드시 문서 내용을 기반으로 할 필요는 없고, 일반적인 질문 형태로 만들어주면 돼.
                """
                prompt = HumanMessage(content=prompt_text)
                response = llm.invoke([prompt])
                return response.content.strip().strip("'\"") # 따옴표 제거
            except Exception as e:
                st.warning(f"❕ 질문 확장 실패: {safe_unicode(str(e))}")
                return query
        return query
    return expand

# --- Streamlit App 실행 부분 ---
# ... (기존과 동일) ...

# openai_api_key 설정, st.set_page_config 등은 그대로 유지

# main 로직
# ...

# if user_query.strip():
#     # query 확장 로직은 개선된 get_query_expander 에 따라 동작
#     expanded_query = query_expander(user_query)
#     if expanded_query != user_query:
#         st.write(f"질문 확장: {expanded_query}") # 사용자에게 확장된 질문을 보여주는 것도 좋음

#     with st.spinner("답변 생성 중..."):
#         try:
#             result = qa_chain.invoke({"query": expanded_query})
#             # ... 이후 로직은 동일 ...
