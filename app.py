import os
import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 🔑 OpenAI API 키 설정
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API 키가 설정되지 않았습니다.")
    st.stop()

# 🌐 사용자 UI
st.image("1.png", width=110)
st.markdown("<h1 style='display:inline-block; margin-left:10px;'>삼성전기 존중노동조합 상담사</h1>", unsafe_allow_html=True)
st.write("안녕하세요! 여기에 질문을 입력하면, 노조 관련 자료에서 자동으로 답변을 제공합니다.")

# 📁 벡터 DB 경로
VECTOR_DB_PATH = "faiss_index"

# 📂 PDF 로딩
def load_documents():
    loader1 = PyPDFLoader("./data/25년 정부 노동정책 주요 아젠다(250627).pdf")
    loader2 = PyPDFLoader("./data/존중노조 노사 정기협의체(250704).pdf")
    documents1 = loader1.load()
    documents2 = loader2.load()
    return documents1 + documents2

# 🧠 벡터 DB 생성
def create_vector_db(texts, embeddings):
    docs_embedded = []
    batch_size = 10

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            embedded = embeddings.embed_documents([d.page_content for d in batch])
            for doc, vec in zip(batch, embedded):
                doc.embedding = vec
                docs_embedded.append(doc)
            time.sleep(1.2)  # Rate limit 완화
        except Exception as e:
            st.warning(f"⚠️ {i}번째 배치 처리 중 오류: {str(e)}")
            time.sleep(5)

    db = FAISS.from_documents(docs_embedded, embeddings)
    db.save_local(VECTOR_DB_PATH)
    return db

# 📌 벡터 DB 불러오기 또는 생성
def load_or_create_vector_db():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    if os.path.exists(VECTOR_DB_PATH):
        try:
            db = FAISS.load_local(VECTOR_DB_PATH, embeddings)
            return db
        except Exception as e:
            st.warning("❌ 기존 벡터 DB 로드 실패. 새로 생성합니다.")
            os.remove(VECTOR_DB_PATH)

    # 새로 생성
    documents = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return create_vector_db(texts, embeddings)

# ✅ 벡터 DB 로딩
db = load_or_create_vector_db()
retriever = db.as_retriever()

# 🤖 챗봇 체인 구성
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
        try:
            result = qa_chain(query)
            st.success(result["result"])

            # 📎 참조 문서 보여주기
            with st.expander("📎 관련 문서 보기"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**문서 {i+1}:** {doc.metadata['source']}")
                    st.write(doc.page_content[:1000])  # 미리보기 제한
        except Exception as e:
            st.error(f"⚠️ 답변 생성 중 오류가 발생했습니다: {str(e)}")
