import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# --- CONFIGURATION ---
# Load OpenAI API key from Streamlit secrets
# Fallback to environment variable if not found in secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, AttributeError):
    openai_api_key = os.getenv("OPENAI_API_KEY")

# Stop the app if the API key is not set
if not openai_api_key:
    st.error("OpenAI API 키가 설정되지 않았습니다. Streamlit secrets에 'OPENAI_API_KEY'를 추가해주세요.")
    st.stop()

# Define paths to your PDF files
PDF_FILES_DIR = "./data/"
PDF_FILES = [
    "2025_government_labor_policy_agenda.pdf",
    "seme_union_meeting_250704.pdf"
]

# --- UI SETUP ---
st.set_page_config(page_title="노조 상담 챗봇", page_icon="🤖")

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stSpinner > div > div {
        border-top-color: #0062ff;
    }
    .stSuccess {
        background-color: #e6f7ff;
        border: 1px solid #91d5ff;
        border-radius: 8px;
        color: #0050b3;
    }
</style>
""", unsafe_allow_html=True)


# App header
st.image("1.png", width=110)
st.markdown("<h1 style='display:inline-block; vertical-align:middle; margin-left:10px; color: #0d1a44;'>삼성전기 존중노동조합 상담사</h1>", unsafe_allow_html=True)
st.write("안녕하세요! 노조 관련 자료를 기반으로 질문에 답변해 드립니다. 아래에 질문을 입력해주세요.")
st.markdown("---")


# --- DATA LOADING AND PROCESSING ---

@st.cache_resource
def load_all_documents(pdf_paths):
    """
    Loads documents from a list of PDF file paths.
    Caches the result to avoid reloading on every interaction.
    """
    all_docs = []
    for path in pdf_paths:
        if os.path.exists(path):
            try:
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
            except Exception as e:
                st.warning(f"'{path}' 파일을 로드하는 중 오류 발생: {e}")
        else:
            st.warning(f"'{path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    return all_docs

@st.cache_resource
def split_documents_into_chunks(documents):
    """
    Splits loaded documents into smaller chunks for processing.
    Caches the result.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    # The content is already a clean string, no need for encode/decode
    return texts

@st.cache_resource
def create_vector_store(_texts, _embedding_model):
    """
    Creates a FAISS vector store from text chunks and an embedding model.
    The '_' prefix in args tells Streamlit to hash the object's contents for caching.
    """
    try:
        vector_store = FAISS.from_documents(_texts, _embedding_model)
        return vector_store
    except Exception as e:
        st.error(f"벡터 DB 생성 중 오류 발생: {e}")
        st.stop()

# --- QA CHAIN SETUP ---

@st.cache_resource
def initialize_qa_chain():
    """
    Initializes all components and builds the RetrievalQA chain.
    This function orchestrates the entire setup process and caches the final chain.
    """
    # 1. Initialize embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # 2. Load documents
    full_pdf_paths = [os.path.join(PDF_FILES_DIR, f) for f in PDF_FILES]
    documents = load_all_documents(full_pdf_paths)
    if not documents:
        st.error("로드할 문서가 없습니다. 'data' 폴더에 PDF 파일이 있는지 확인해주세요.")
        st.stop()

    # 3. Split documents
    text_chunks = split_documents_into_chunks(documents)

    # 4. Create Vector DB
    db = create_vector_store(text_chunks, embeddings)

    # 5. Create Retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 6. Initialize LLM and create QA chain
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# --- MAIN APPLICATION LOGIC ---

# Initialize the QA chain
try:
    qa_chain = initialize_qa_chain()
except Exception as e:
    st.error(f"챗봇 초기화 중 심각한 오류가 발생했습니다: {e}")
    st.stop()


# Get user input
query = st.text_input(
    "무엇이 궁금하신가요?",
    placeholder="예: 7월 정기협의 주요 의제는 무엇인가요?",
    key="query_input"
)

if query:
    with st.spinner("답변을 생성하고 있습니다... 잠시만 기다려주세요."):
        try:
            # Run the QA chain with the user's query
            result = qa_chain.invoke({"query": query})

            # Display the answer
            st.success(result["result"])

            # Display source documents in an expander
            with st.expander("📎 답변 근거 문서 보기"):
                for i, doc in enumerate(result["source_documents"]):
                    source_name = os.path.basename(doc.metadata.get('source', '알 수 없는 출처'))
                    st.markdown(f"**문서 {i+1}:** `{source_name}` (페이지: {doc.metadata.get('page', 'N/A') + 1})")
                    # Display a snippet of the page content
                    st.write(f'"{doc.page_content.strip()[:500]}..."')
                    st.markdown("---")

        except Exception as e:
            st.error(f"답변을 생성하는 중 오류가 발생했습니다: {e}")

