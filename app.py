import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import os

# OpenAI API 키 설정 (secrets or 환경변수 활용)
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

# 기본 세팅
st.set_page_config(page_title="SEMunion Chatbot", layout="wide")

# 로고 및 제목 (좌측 상단 정렬)
st.markdown(
    """
    <div style='display: flex; align-items: center; margin-bottom: 20px;'>
        <img src='https://raw.githubusercontent.com/SHIN-HOON-SIK/semunion-chatbot/main/1.png' width='48' style='margin-right: 15px;'/>
        <h1 style='margin: 0;'>삼성전기 존중노동조합 상담사</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# 인사말
st.write("안녕하세요! 여기에 질문을 입력하면, 노조 관련 자료에서 자동으로 답변을 제공합니다.")

# 파일 로딩 (문서 위치는 `/data/sample.txt`로 가정)
loader = PyPDFLoader("./data/삼성전기_상담자료_테스트.pdf")
documents = loader.load()

# 문서 분할 및 벡터화
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever()
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 사용자 질문 입력
question = st.text_input("질문을 입력하세요:")
if question:
    with st.spinner("답변 생성 중..."):
        result = qa_chain.run(question)
        st.markdown(f"**답변:** {result}")

# 하단 연락처 / 사업자 정보
st.markdown("---")
st.markdown(
    """
    <div style='font-size: 12px; color: gray;'>
        수원시 영통구 매영로 159번길 19 광교 더 퍼스트 지식산업센터<br>
        사업자 등록번호 133-82-71927 / 대표 : 신훈식 / 대표번호 : 010-9496-6517 / 이메일 : hoonsik79@hanmail.net
    </div>
    """,
    unsafe_allow_html=True
)
