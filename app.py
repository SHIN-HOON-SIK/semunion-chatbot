import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 🔐 API 키 로딩
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
    st.stop()

# 📘 문서 로딩 함수
def load_documents():
    loaders = []
    for filename in os.listdir("./data"):
        if filename.endswith(".pdf"):
            loaders.append(PyPDFLoader(f"./data/{filename}"))
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents

# 📄 텍스트 분할
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

# 🧠 벡터 DB 생성 함수 (에러 방지 및 인코딩 처리)
@st.cache_data(show_spinner="📊 벡터 DB 생성 중...")
def create_vector_db(docs, embedding_model):
    try:
        contents = [doc.page_content for doc in docs]
        safe_contents = [
            str(text).encode("utf-8", errors="ignore").decode("utf-8")
            for text in contents
        ]
        vectors = embedding_model.embed_documents(safe_contents)
        return FAISS.from_texts(safe_contents, embedding_model)
    except Exception as e:
        st.error(f"❌ 임베딩 생성 중 오류 발생: {e}")
        st.stop()

# 🔧 전체 파이프라인
def load_or_create_vector_db():
    documents = load_documents()
    texts = split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return create_vector_db(texts, embeddings)

# 🧠 LLM QA 체인 생성
@st.cache_data(show_spinner="🤖 챗봇 로딩 중...")
def get_qa_chain(db):
    retriever = db.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# 💻 Streamlit UI
st.image("1.png", width=110)
st.markdown("<h1 style='display:inline-block; margin-left:10px;'>삼성전기 존중노동조합 상담사</h1>", unsafe_allow_html=True)
st.write("안녕하세요! 여기에 질문을 입력하면, 노조 관련 자료에서 자동으로 답변을 제공합니다.")

# 📦 DB 및 체인 준비
db = load_or_create_vector_db()
qa_chain = get_qa_chain(db)

# 🔍 사용자 질문 입력
query = st.text_input("무엇이 궁금하신가요?", placeholder="예: 7월 정기협의 주요 의제는 무엇인가요?")

# 💬 응답 출력
if query:
    with st.spinner("답변 생성 중..."):
        result = qa_chain(query)
        st.success(result["result"])

        # 🔎 참고 문서
        with st.expander("📎 관련 문서 보기"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**문서 {i+1}:** {doc.metadata['source']}")
                st.write(doc.page_content[:1000])  # 1000자 미리보기

# 🔄 텍스트 분할 전 처리
docs_raw = documents1 + documents2

# 한글 포함 문서의 유니코드 오류 방지
for doc in docs_raw:
    try:
        doc.page_content = doc.page_content.encode('utf-8').decode('utf-8')
    except UnicodeEncodeError:
        st.warning("⚠️ 문서 인코딩 중 문제가 발생했습니다.")
        st.stop()

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(docs_raw)

