import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# OpenAI API 키
openai_api_key = st.secrets["OPENAI_API_KEY"]

# 제목과 설명
st.image("1.png", width=120)
st.markdown("## 삼성전기 존중노동조합 상담사")
st.write("안녕하세요! 여기에 질문을 입력하면, 노조 관련 자료에서 자동으로 답변을 제공합니다.")

# PDF 문서 로딩
loader1 = PyPDFLoader("./data/25년 정부 노동정책 주요 아젠다(250627).pdf")
loader2 = PyPDFLoader("./data/존중노조 노사 정기협의체(250704).pdf")

docs1 = loader1.load()
docs2 = loader2.load()
documents = docs1 + docs2

# 문서 쪼개기
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 벡터 DB 생성
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(texts, embeddings)

# 질의응답 체인 생성
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    retriever=db.as_retriever()
)

# 사용자 입력
query = st.text_input("질문을 입력하세요")

# 답변 출력
if query:
    result = qa.run(query)
    st.write("🧠 답변:", result)

# 하단 정보
st.markdown("""
---
**수원시 영통구 매영로 159번길 19 광교 더 퍼스트 지식산업센터**  
사업자등록번호: 133-82-71927  
대표: 신훈식 | 대표번호: 010-9496-6517 | 이메일: hoonsik79@hanmail.net
""")
