# union_chatbot_app.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# 페이지 설정
st.set_page_config(page_title="삼성전기 존중노동조합 상담사", layout="centered")

# 💡 전체 배경 흰색으로 설정하는 CSS
st.markdown(
    """
    <style>
        .stApp {
            background-color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# OpenAI API 키 설정
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, AttributeError):
    openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OpenAI API 키가 설정되지 않았습니다. Streamlit secrets 또는 환경변수를 확인해주세요.")
    st.stop()

# PDF 파일 경로 설정
BASE_DIR = Path(__file__).parent
PDF_FILES_DIR = BASE_DIR / "data"
PDF_FILES = [
    "policy_agenda_250627.pdf",
    "union_meeting_250704.pdf"
]

# UI 구성
st.markdown("""
<style>
    .stApp { background-color: white; }
    .stSpinner > div > div { border-top-color: #0062ff; }
    .stSuccess {
        background-color: #e6f7ff;
        border: 1px solid #91d5ff;
        border-radius: 8px;
        color: #0050b3;
    }
</style>
""", unsafe_allow_html=True)

st.image("1.png", width=300)

st.markdown("""
    <style>
    .footer-left {
        position: fixed;
        bottom: 10px;
        left: 10px;
        font-size: 12px;
        color: #555;
        line-height: 1.5;
        z-index: 100;
    }
    </style>

    <div class="footer-left">
        수원시 영통구 매영로 159번길 19, 광교 더 퍼스트 지식산업센터<br>
        사업자등록번호: 133-82-71927 ｜ 대표: 신훈식 ｜ 대표번호: 010-9496-6517<br>
        이메일: <a href="mailto:hoonsik79@hanmail.net">hoonsik79@hanmail.net</a>
    </div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='display:inline-block; vertical-align:middle; margin-left:10px; color: #0d1a44;'>삼성전기 존중노동조합 상담사</h1>", unsafe_allow_html=True)
st.write("안녕하세요!노조 집행부에서 업로드 한 자료에 기반하여 노조 및 회사 관련 질문에 답변해 드립니다. 아래에 질문을 입력해 주세요.")

# 문서 로딩 및 처리 함수
@st.cache_resource
def load_all_documents(pdf_paths):
    all_docs = []
    for path in pdf_paths:
        if path.exists():
            try:
                loader = PyPDFLoader(str(path))
                all_docs.extend(loader.load())
            except Exception as e:
                st.warning(f"'{path.name}' 파일을 로드하는 중 오류 발생: {e}")
        else:
            st.warning(f"'{path.name}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    return all_docs

@st.cache_resource
def split_documents_into_chunks(_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return text_splitter.split_documents(_documents)

@st.cache_resource
def create_vector_store(_texts, _embedding_model):
    try:
        return FAISS.from_documents(_texts, _embedding_model)
    except Exception as e:
        st.error(f"벡터 DB 생성 중 오류 발생: {e}")
        st.stop()

# 질의응답 체인 구성
@st.cache_resource
def initialize_qa_chain():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    full_pdf_paths = [PDF_FILES_DIR / fname for fname in PDF_FILES]
    documents = load_all_documents(full_pdf_paths)
    if not documents:
        st.error("❌ 로드할 문서가 없습니다. 'data' 폴더에 PDF 파일이 있는지 확인해주세요.")
        st.stop()
    text_chunks = split_documents_into_chunks(documents)
    db = create_vector_store(text_chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# 앱 실행
try:
    qa_chain = initialize_qa_chain()
except Exception as e:
    st.error(f"챗봇 초기화 중 오류 발생: {e}")
    st.stop()

query = st.text_input(
    "[무엇이든 물어보세요.]",
    placeholder="여기에 질문을 입력해 주세요",
    key="query_input"
)

if query:
    with st.spinner("답변을 생성하고 있습니다... 잠시만 기다려주세요."):
        try:
            result = qa_chain.invoke({"query": query})
            answer_text = result["result"].strip()

            if not answer_text or ("정보" in answer_text and "없" in answer_text):
                st.info("죄송하지만 제공된 정보에는 해당 내용이 포함되어 있지 않습니다.")
            else:
                st.success(answer_text)

            with st.expander("📄 답변 근거 문서 보기"):
                for i, doc in enumerate(result["source_documents"]):
                    source_name = Path(doc.metadata.get('source', '알 수 없는 출처')).name
                    page = doc.metadata.get('page')
                    page_number = page + 1 if isinstance(page, int) else "알 수 없음"
                    st.markdown(f"**문서 {i+1}:** `{source_name}` (페이지: {page_number})")
                    st.write(f'"{doc.page_content.strip()[:500]}..."')
                    st.markdown("---")
        except Exception as e:
            st.error(f"❌ 답변 생성 중 오류 발생: {e}")
