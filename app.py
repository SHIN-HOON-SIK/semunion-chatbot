import os
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 🔑 OpenAI API 키 설정
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API 키가 설정되지 않았습니다. secrets.toml 또는 환경 변수에 키를 등록하세요.")
    st.stop()

# ✅ 텍스트 정제 함수 (임베딩 오류 방지)
def sanitize(text):
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

# 📂 PDF 문서 로딩
@st.cache_resource
def load_documents():
    loaders = [
        PyPDFLoader("./data/labor_policy_agenda_250627.pdf"),
        PyPDFLoader("./data/samsung_me_union_meeting_250704.pdf"),
        PyPDFLoader("./data/external_audit_250204.pdf")
    ]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents

# 🧠 벡터 DB 생성
@st.cache_resource(show_spinner="📚 문서 임베딩 중...")
def create_vector_db():
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)
    cleaned_texts = [sanitize(t.page_content) for t in texts if t.page_content.strip()]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_texts(cleaned_texts, embeddings)

# 🤖 QA 체인 구성
@st.cache_resource
def get_qa_chain():
    vectorstore = create_vector_db()
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# 🖼️ 상단 UI
st.image("1.png", width=110)
st.markdown("<h1 style='display:inline-block; margin-left:10px;'>삼성전기 존중노동조합 상담사</h1>", unsafe_allow_html=True)
st.write("노조 관련 문서 기반으로 자동 답변을 제공하는 상담 챗봇입니다.")

# 💬 사용자 질문 입력
query = st.text_input("무엇이 궁금하신가요?", placeholder="예: 외부 감사 준비 항목은?")

if query:
    with st.spinner("🤖 답변 생성 중..."):
        qa = get_qa_chain()
        result = qa(query)
        st.success(result["result"])

        with st.expander("📎 관련 문서 보기"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**문서 {i+1}:** {doc.metadata.get('source', '')}")
                st.write(doc.page_content[:1000])
