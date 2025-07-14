import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ✅ OpenAI API 키 불러오기
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API 키가 설정되지 않았습니다.")
    st.stop()

# ✅ 문서 로딩 함수 (PDF 2개)
@st.cache_resource
def load_documents():
    loader1 = PyPDFLoader("./data/25년 정부 노동정책 주요 아젠다(250627).pdf")
    loader2 = PyPDFLoader("./data/존중노조 노사 정기협의체(250704).pdf")
    docs = loader1.load() + loader2.load()

    # 🔐 한글 인코딩 오류 방지
    for doc in docs:
        try:
            doc.page_content = doc.page_content.encode('utf-8').decode('utf-8')
        except UnicodeEncodeError:
            st.warning("⚠️ 일부 문서에서 인코딩 오류가 발생했습니다.")
            st.stop()

    return docs

# ✅ 벡터 DB 생성 함수
@st.cache_resource
def load_or_create_vector_db():
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_documents(texts, embeddings)

# ✅ QA 체인 생성 함수
@st.cache_resource
def get_qa_chain():
    db = load_or_create_vector_db()
    retriever = db.as_retriever()
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# ✅ Streamlit 사용자 UI
st.image("1.png", width=110)
st.markdown("<h1 style='display:inline-block; margin-left:10px;'>삼성전기 존중노동조합 상담사</h1>", unsafe_allow_html=True)
st.write("안녕하세요! 여기에 질문을 입력하면, 노조 관련 자료에서 자동으로 답변을 제공합니다.")

# ✅ 사용자 질문 입력
query = st.text_input("무엇이 궁금하신가요?", placeholder="예: 7월 정기협의 주요 의제는 무엇인가요?")

# ✅ 응답 처리
if query:
    qa_chain = get_qa_chain()
    with st.spinner("✍️ 답변 생성 중..."):
        result = qa_chain(query)
        st.success(result["result"])

        # 📎 출처 문서 표시
        with st.expander("📚 관련 문서 보기"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**문서 {i+1}:** {doc.metadata['source']}")
                st.write(doc.page_content[:1000])
