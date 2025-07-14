import streamlit as st

st.set_page_config(page_title="SEMunion Chatbot", layout="centered")

st.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <img src='https://raw.githubusercontent.com/SHIN-HOON-SIK/semunion-chatbot/main/1.png' width='80' style='vertical-align: middle; margin-right: 10px;'/>
        <span style='font-size: 32px; font-weight: bold; vertical-align: middle;'>삼성전기 존중노동조합 상담사</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("안녕하세요! 여기에 질문을 입력하면, 노조 관련 자료에서 자동으로 답변을 제공합니다.")
