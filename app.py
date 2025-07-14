import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import tempfile
import shutil

# 1. 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 2. LangChain 모델 정의
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
chain = load_qa_chain(llm, chain_type="stuff")

# 3. 인코딩 안전 텍스트 처리 함수
def safe_text(text):
    return text.encode('utf-8', errors='ignore').decode('utf-8')

# 4. 벡터 DB 생성 함수 (캐싱 포함)
@st.cache_resource(show_spinner="임베딩 로딩 중...")
def load_or_create_vector_db():
    if os.path.exists("data/index.faiss") and os.path.exists("data/index.pkl"):
        return FAISS.load_local("data/index", embedding_model, allow_dangerous_deserialization=True)
    else:
        docs = []
        for filename in os.listdir("data"):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join("data", filename))
                docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs_split = splitter.split_documents(docs)
        contents = [safe_text(doc.page_content) for doc in docs_split]

        vectors = embedding_model.embed_documents(contents)
        db = FAISS.from_documents(docs_split, embedding_model)
        db.save_local("data/index")
        return db

# 5. Streamlit UI 설정
st.set_page_config(page_title="삼성전기 존중노동조합 상담사", page_icon="🤖")
st.markdown("""
    # 삼성전기 존중노동조합 상담사
    안녕하세요! 여기에 질문을 입력하면, 노조 관련 자료에서 자동으로 답변을 제공합니다.
""")

# 6. 사용자 질문 입력 및 응답 처리
query = st.text_input("질문을 입력하세요", placeholder="예: 상조 지원 대상이 누군가요?")
if query:
    db = load_or_create_vector_db()
    docs = db.similarity_search(query, k=3)
    response = chain.run(input_documents=docs, question=query)
    st.markdown("---")
    st.subheader("답변")
    st.write(response)
