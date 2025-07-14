// union_search_app: MVP 노조 검색 도우미

// 📁 server/index.js (Node.js 백엔드)
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { Configuration, OpenAIApi } = require('openai');

const app = express();
app.use(cors());
app.use(bodyParser.json());

const configuration = new Configuration({
  apiKey: 'YOUR_OPENAI_API_KEY', // OpenAI API 키 설정
});
const openai = new OpenAIApi(configuration);

// 임시 문서 데이터
const documents = [
  {
    id: 1,
    title: "2025-03-07 상조회 변경 공지",
    content: "2025년 3월부터 가족관계증명서 제출이 의무화되며 계부모는 지원 대상에 포함되지 않습니다."
  },
  {
    id: 2,
    title: "2024-12-12 복지제도 안내",
    content: "복지 포인트는 연간 50만 원 한도로 지급되며, 미사용 시 이월되지 않습니다."
  }
];

app.post('/ask', async (req, res) => {
  const { question } = req.body;

  // 단순 키워드 검색 (향후 FAISS로 대체 예정)
  const matched = documents.filter(doc => doc.content.includes(question) || doc.title.includes(question));
  const context = matched.map(d => `${d.title}: ${d.content}`).join('\n');

  try {
    const completion = await openai.createChatCompletion({
      model: 'gpt-4',
      messages: [
        { role: 'system', content: '다음 문서를 기반으로 노조원의 질문에 답하세요:' },
        { role: 'user', content: `문서:\n${context}\n\n질문:\n${question}` }
      ]
    });
    res.json({ answer: completion.data.choices[0].message.content });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3001, () => console.log('Server running on port 3001'));


// 📁 client/pages/index.js (Next.js 프론트)
import { useState } from 'react';

export default function Home() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');

  const ask = async () => {
    const res = await fetch('http://localhost:3001/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await res.json();
    setAnswer(data.answer);
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>노조 검색 도우미</h1>
      <input
        type="text"
        value={question}
        onChange={e => setQuestion(e.target.value)}
        placeholder="예: 상조 기준이 뭐예요?"
        style={{ width: '80%', padding: '0.5rem' }}
      />
      <button onClick={ask} style={{ marginLeft: '1rem' }}>검색</button>
      <div style={{ marginTop: '2rem' }}>
        <h3>답변:</h3>
        <p>{answer}</p>
      </div>
    </div>
  );
}
