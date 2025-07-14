import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 🔑 OpenAI API 키 (Streamlit secrets 또는 환경 변수 사용)
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API 키가 설정되지 않았습니다.")
    st.stop()

# 🌐 사용자 UI
st.image("1.png", width=110)
st.markdown("<h1 style='display:inline-block; margin-left:10px;'>삼성전기 존중노동조합 상담사</h1>", unsafe_allow_html=True)
st.write("안녕하세요! 여기에 질문을 입력하면, 노조 관련 자료에서 자동으로 답변을 제공합니다.")

# 📂 PDF 로딩 (data 폴더 내 2개 파일)
loader1 = PyPDFLoader("./data/25년 정부 노동정책 주요 아젠다(250627).pdf")
loader2 = PyPDFLoader("./data/존중노조 노사 정기협의체(250704).pdf")
documents1 = loader1.load()
documents2 = loader2.load()
documents = documents1 + documents2

# 🔄 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# 🧠 임베딩 + 벡터 DB
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

# 🤖 챗봇 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 💬 사용자 질문 입력창
query = st.text_input("무엇이 궁금하신가요?", placeholder="예: 7월 정기협의 주요 의제는 무엇인가요?")

if query:
    with st.spinner("답변 생성 중..."):
        result = qa_chain(query)
        st.success(result["result"])

        # 🔍 참조 문서 보여주기
        with st.expander("📎 관련 문서 보기"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**문서 {i+1}:** {doc.metadata['source']}")
                st.write(doc.page_content[:1000])  # 1,000자까지만 미리보기
