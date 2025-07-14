import os import streamlit as st from langchain_community.document_loaders import PyPDFLoader from langchain.text_splitter import RecursiveCharacterTextSplitter from langchain_community.vectorstores import FAISS from langchain.embeddings.openai import OpenAIEmbeddings from langchain.chat_models import ChatOpenAI from langchain.chains import RetrievalQA

🔑 OpenAI API 키 설정

openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY") if not openai_api_key: st.error("OpenAI API 키가 설정되지 않았습니다.") st.stop()

🌐 사용자 UI

st.image("1.png", width=110) st.markdown("<h1 style='display:inline-block; margin-left:10px;'>삼성전기 존중노동조합 상담사</h1>", unsafe_allow_html=True) st.write("안녕하세요! 여기에 질문을 입력하면, 노조 관련 자료에서 자동으로 답변을 제공합니다.")

📂 PDF 로딩

@st.cache_resource def load_documents(): loader1 = PyPDFLoader("./data/2025_government_labor_policy_agenda.pdf") loader2 = PyPDFLoader("./data/seme_union_meeting_250704.pdf") documents1 = loader1.load() documents2 = loader2.load() return documents1 + documents2

🔄 텍스트 분할

@st.cache_resource def preprocess_documents(documents): text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) texts = text_splitter.split_documents(documents) contents = [t.page_content.strip().encode("utf-8", errors="ignore").decode("utf-8") for t in texts if t.page_content.strip()] return contents

🧠 벡터 DB 생성

@st.cache_resource def create_vector_db(contents, embedding_model): return FAISS.from_texts(contents, embedding_model)

🧠 전체 임베딩 처리

@st.cache_resource def load_or_create_vector_db(): documents = load_documents() contents = preprocess_documents(documents) embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key) return create_vector_db(contents, embedding_model)

🤖 챗봇 생성

@st.cache_resource def create_qa_chain(): db = load_or_create_vector_db() retriever = db.as_retriever() return RetrievalQA.from_chain_type( llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0), chain_type="stuff", retriever=retriever, return_source_documents=True )

qa_chain = create_qa_chain()

💬 사용자 질문 입력

query = st.text_input("무엇이 궁금하신가요?", placeholder="예: 7월 정기협의 주요 의제는 무엇인가요?")

if query: with st.spinner("답변 생성 중..."): result = qa_chain(query) st.success(result["result"])

# 🔍 참조 문서 보여주기
    with st.expander("📎 관련 문서 보기"):
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**문서 {i+1}:** {doc.metadata['source']}")
            st.write(doc.page_content[:1000])

