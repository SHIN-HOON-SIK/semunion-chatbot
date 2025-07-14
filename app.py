import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 📍 벡터 DB 경로
VECTOR_DB_PATH = "faiss_index"

# 🔑 OpenAI API 키 설정
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API 키가 설정되지 않았습니다.")
    st.stop()

# 💡 캐시: PDF 로딩 및 분할
@st.cache_resource(show_spinner=False)
def load_documents():
    loader1 = PyPDFLoader("./data/25년 정부 노동정책 주요 아젠다(250627).pdf")
    loader2 = PyPDFLoader("./data/존중노조 노사 정기협의체(250704).pdf")
    docs = loader1.load() + loader2.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

# 💾 캐시: FAISS 벡터 DB 저장 또는 로딩
@st.cache_resource(show_spinner=False)
def load_or_create_vector_db():
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        return create_vector_db(texts, embeddings)

# 💾 FAISS 벡터 DB 생성
def create_vector_db(docs, embedding_model):
    contents = [doc.page_content for doc in docs]
    try:
        vectors = embedding_model.embed_documents(contents)
    except Exception as e:
        st.error(f"임베딩 오류: {str(e)}")
        raise e

    db = FAISS.from_embeddings(vectors, docs, embedding_model)
    db.save_local(VECTOR_DB_PATH)
    return db

# 📦 문서 임베딩
texts = load_documents()
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = load_or_create_vector_db()
retriever = db.as_retriever()

# 🤖 챗봇 QA 체인
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 🌐 UI 구성
st.image("1.png", width=110)
st.markdown("<h1 style='display:inline-block; margin-left:10px;'>삼성전기 존중노동조합 상담사</h1>", unsafe_allow_html=True)
st.write("안녕하세요! 여기에 질문을 입력하면, 노조 관련 자료에서 자동으로 답변을 제공합니다.")

query = st.text_input("무엇이 궁금하신가요?", placeholder="예: 7월 정기협의 주요 의제는 무엇인가요?")

if query:
    with st.spinner("답변 생성 중..."):
        result = qa_chain(query)
        st.success(result["result"])

        with st.expander("📎 관련 문서 보기"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**문서 {i+1}:** {doc.metadata['source']}")
                st.write(doc.page_content[:1000])
