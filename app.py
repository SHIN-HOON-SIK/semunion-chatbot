import os import sys import io import streamlit as st from langchain_community.document_loaders import PyPDFLoader from langchain.text_splitter import RecursiveCharacterTextSplitter from langchain_community.vectorstores import FAISS from langchain.embeddings.openai import OpenAIEmbeddings from langchain.chat_models import ChatOpenAI from langchain.chains import RetrievalQA

✅ 시스템 출력 인코딩 설정 (UnicodeEncodeError 방지)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

✅ 전처리 함수 (텍스트 내 특수문자나 비ASCII 문자 정리)

def clean_text(text): return text.encode("utf-8", errors="ignore").decode("utf-8")

🔑 OpenAI API 키

openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY") if not openai_api_key: st.error("OpenAI API 키가 설정되지 않았습니다.") st.stop()

🌐 사용자 UI

st.image("1.png", width=110) st.markdown("<h1 style='display:inline-block; margin-left:10px;'>삼성전기 존중노동조합 상담사</h1>", unsafe_allow_html=True) st.write("안녕하세요! 여기에 질문을 입력하면, 노조 관련 자료에서 자동으로 답변을 제공합니다.")

📂 PDF 문서 로딩

loader1 = PyPDFLoader("./data/25_government_agenda.pdf") loader2 = PyPDFLoader("./data/respect_union_agenda.pdf") loader3 = PyPDFLoader("./data/external_audit_preparation.pdf")  # 새 문서 추가

documents = loader1.load() + loader2.load() + loader3.load()

🔄 텍스트 분할 + 전처리 적용

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) texts = text_splitter.split_documents(documents) contents = [clean_text(doc.page_content) for doc in texts]

🧠 임베딩 + 벡터 DB 생성

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) db = FAISS.from_texts(contents, embeddings) retriever = db.as_retriever()

🤖 챗봇 생성

qa_chain = RetrievalQA.from_chain_type( llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0), chain_type="stuff", retriever=retriever, return_source_documents=True )

💬 사용자 입력

query = st.text_input("무엇이 궁금하신가요?", placeholder="예: 외부 회계감사 준비 내용은?")

if query: with st.spinner("답변 생성 중..."): result = qa_chain(query) st.success(result["result"])

# 🔍 참조 문서 미리보기
    with st.expander("📎 관련 문서 보기"):
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**문서 {i+1}:** {doc.metadata['source']}")
            st.write(doc.page_content[:1000])

