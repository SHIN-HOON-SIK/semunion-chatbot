# app.py
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

# OpenAI API 키 입력
openai.api_key = "YOUR_OPENAI_API_KEY"  # 실제 키로 바꿔주세요

# 임시 문서 데이터
documents = [
    {
        "title": "2025-03-07 상조회 변경 공지",
        "content": "2025년 3월부터 가족관계증명서 제출이 의무화되며 계부모는 지원 대상에 포함되지 않습니다."
    },
    {
        "title": "2024-12-12 복지제도 안내",
        "content": "복지 포인트는 연간 50만 원 한도로 지급되며, 미사용 시 이월되지 않습니다."
    }
]

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    # 간단한 키워드 기반 문서 필터링
    relevant_docs = [
        f"{doc['title']}:\n{doc['content']}"
        for doc in documents
        if question in doc['content'] or question in doc['title']
    ]
    context = "\n\n".join(relevant_docs) or "관련 문서를 찾을 수 없습니다."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "다음 문서를 기반으로 노조원의 질문에 답하세요."},
                {"role": "user", "content": f"문서:\n{context}\n\n질문:\n{question}"}
            ]
        )
        return jsonify({"answer": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
