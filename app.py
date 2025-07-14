import streamlit as st

st.set_page_config(page_title="SEMUNION 상담사 ", layout="centered")

st.markdown(
    """
    <div style='text-align: center; margin-bottom: 10px;'>
        <img src='https://raw.githubusercontent.com/SHIN-HOON-SIK/semunion-chatbot/main/2.png' width='240'/>
    </div>
    """,
    unsafe_allow_html=True

  st.title("삼성전기 존중노동조합 상담사")
st.write("안녕하세요! 여기에 질문을 입력하면, 노조 및 회사 생활에 관련된 내용에 자동으로 답변을 제공합니다.")
