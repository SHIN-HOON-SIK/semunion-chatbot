import streamlit as st

st.set_page_config(page_title="SEMunion Chatbot", layout="centered")

st.markdown("""
    <div style='text-align: left; margin-bottom: 10px;'>
        <img src='https://raw.githubusercontent.com/SHIN-HOON-SIK/semunion-chatbot/main/1.png' width='400'/>
    </div>
""", unsafe_allow_html=True)

st.markdown(
    """
    <hr style='margin-top: 50px; margin-bottom: 10px;'>

    <div style='font-size: 13px; color: #888; line-height: 1.6;'>
        주소: 수원시 영통구 매영로 159번길 19, 광교 더 퍼스트 지식산업센터<br>
        사업자등록번호: 133-82-71927 | 대표: 신훈식<br>
        대표번호: 010-9496-6517 | 이메일: hoonsik79@hanmail.net
    </div>
    """,
    unsafe_allow_html=True
)

st.title("※노조 전문 상담사")
st.write("안녕하세요! 여기에 질문을 입력하면, 노조 및 회사생활 관련 내용에 대해 자동으로 답변을 제공합니다.")
