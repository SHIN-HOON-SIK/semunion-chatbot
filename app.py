import os
import sys
import io
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ✅ 시스템 출력 인코딩 설정 (UnicodeEncodeError 방지)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ✅ 전처리 함수 (텍스트 내 특수문자나 비ASCII 문자 정리 및 인코딩 강화)
def clean_text(text):
    if isinstance(text, str):
        # UTF-8로 인코딩하되, 오류가 발생하면 해당 문자를 무시합니다.
        # 다시 UTF-8로 디코딩하여 유니코드 문자열 상태를 유지합니다.
        cleaned = text.encode("utf-8", errors="ignore").decode("utf-8")
        # 추가적으로, API로 전달될 때 문제가 될 수 있는 비표준 공백 문자 등을 제거하고
        # 연속된 공백을 단일 공백으로 처리합니다.
        return ' '.join(cleaned.split()).strip()
    return "" # 문자열이 아닌 경우 빈 문자열 반환

# 🔑 OpenAI API 키 확인
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API 키가 설정되지 않았습니다. 'OPENAI_API_KEY' 환경 변수 또는 Streamlit secrets에 키를 설정해주세요.")
    st.stop()

# 🌐 사용자 UI (요청하신 내용 그대로 적용)
st.image("1.png", width=110)
st.markdown("<h1 style='display:inline-block; margin-left:10px;'>삼성전기 존중노조 상담사</h1>", unsafe_allow_html=True)
st.write("노조 관련 문서 기반으로 자동 답변을 제공합니다. 궁금한 내용을 입력해보세요.")


# 📂 PDF 문서 로딩 및 전처리 함수 (성능 최적화를 위한 캐싱 적용)
@st.cache_data(show_spinner="PDF 문서 로딩 및 텍스트 전처리 중...")
def load_and_process_documents(file_paths):
    all_documents = []
    for path in file_paths:
        try:
            loader = PyPDFLoader(path)
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            st.error(f"오류: '{path}' 파일을 로드하는 데 실패했습니다. 파일 경로를 확인해주세요. ({e})")
            continue

    if not all_documents:
        st.error("처리할 문서가 없습니다. PDF 파일이 제대로 로드되었는지 확인해주세요.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(all_documents)

    # 모든 텍스트 청크에 clean_text 적용
    cleaned_contents = [clean_text(doc.page_content) for doc in texts]

    return cleaned_contents, texts

# 🧠 임베딩 + 벡터 DB 생성 함수 (성능 최적화를 위한 캐싱 적용)
@st.cache_resource(show_spinner="벡터 데이터베이스 생성 중...")
def create_vector_db_and_retriever(cleaned_contents, openai_key):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        db = FAISS.from_texts(cleaned_contents, embeddings)
        return db.as_retriever()
    except Exception as e:
        st.error(f"벡터 데이터베이스 생성 중 오류가 발생했습니다: {e}. API 키와 네트워크 연결을 확인해주세요.")
        st.stop()

# ---
# 메인 로직 시작
# ---

# 로드할 PDF 파일 경로 목록
pdf_file_paths = [
    "./data/25_government_agenda.pdf",
    "./data/respect_union_agenda.pdf",
    "./data/external_audit_preparation.pdf"
]

# 1. 문서 로드 및 전처리 (캐싱 적용)
cleaned_contents, original_texts = load_and_process_documents(pdf_file_paths)

# 2. 벡터 DB 및 리트리버 생성 (캐싱 적용)
retriever = create_vector_db_and_retriever(cleaned_contents, openai_api_key)

# 3. 챗봇 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-4o"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 💬 사용자 입력
query = st.text_input("무엇이 궁금하신가요?", placeholder="예: 외부 회계감사 준비 내용은?")

if query:
    with st.spinner("답변 생성 중..."):
        try:
            result = qa_chain({"query": query})
            st.success(result["result"])

        except Exception as e:
            st.error(f"답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요. ({e})")

    # 🔍 참조 문서 미리보기
    with st.expander("📎 관련 문서 보기"):
        if "source_documents" in result and result["source_documents"]:
            for i, doc in enumerate(result["source_documents"]):
                source_info = doc.metadata.get('source', '알 수 없음')
                st.markdown(f"**문서 {i+1}:** `{source_info}`")
                st.write(doc.page_content[:1000])
        else:
            st.info("이 질문에 대한 관련 문서를 찾을 수 없습니다.")
