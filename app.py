import streamlit as st

st.set_page_config(page_title="SEMunion Chatbot", layout="centered")

st.markdown("""
    <div style='text-align: center; margin-bottom: 10px;'>
        <img src='https://raw.githubusercontent.com/SHIN-HOON-SIK/semunion-chatbot/main/1.png' width='320'/>
    </div>
""", unsafe_allow_html=True)

st.title("노조 전문 상담사")
st.write("안녕하세요! 여기에 질문을 입력하면, 노조 관련 자료에서 자동으로 답변을 제공합니다.")
