import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

# --- 페이지 설정 ---
st.set_page_config(page_title="SEMunion Chatbot", layout="wide")

# --- 상단 로고 + 제목 (좌측 정렬) ---
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("1.png", width=70)  # 로고 이미지 (로컬 파일 경로)
with col2:
    st.title("♥노조 전문 상담사")

st.markdown("안녕하세요! 여기에 질문을 입력하면, 노조 관련 자료에서 자동으로 답변을 제공합니다.")

# --- 하단 사업자 정보 ---
with st.sidebar:
    st.markdown("---")
    st.caption("📍 수원시 영통구 매영로 159번길 19, 광교 더 퍼스트 지식산업센터")
    st.caption("📄 사업자등록번호: 133-82-71927")
    st.caption("👤 대표: 신훈식")
    st.caption("📞 010-9496-6517")
    st.caption("📧 hoonsik79@hanmail.net")

# --- 예시 안내 ---
st.info("💬 예시 질문: '상조회 신청은 어떻게 하나요?'")

# --- 문서 로딩 및 임베딩 ---
@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader("sample.pdf")  # 테스트용 PDF 파일명
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = load_vectorstore()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# --- 사용자 질문 입력 ---
question = st.text_input("질문을 입력하세요:")

# --- 응답 출력 ---
if question:
    with st.spinner("답변 생성 중..."):
        answer = qa.run(question)
        st.write("🧠 답변:", answer)
