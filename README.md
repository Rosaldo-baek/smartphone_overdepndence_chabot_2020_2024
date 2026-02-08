# 스마트폰 과의존 실태조사 RAG 챗봇

2020~2024년 스마트폰 과의존 실태조사 보고서 분석 시스템입니다.

## 주요 기능

- **연도별 분석**: 2020~2024년 5개년 보고서 데이터 검색
- **멀티연도 비교**: 여러 연도 데이터를 표 형식으로 비교
- **대화 맥락 유지**: 후속 질문 지원
- **실시간 진행 상태**: 분석 → 검색 → 생성 → 검증 단계 표시

## 로컬 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. API 키 설정 (택 1)
# 방법 A: 환경변수
export OPENAI_API_KEY="sk-your-key"

# 방법 B: 파일
echo "sk-your-key" > openai_api_for_rag_test.txt

# 3. 실행
streamlit run streamlit_app.py
```

## Streamlit Share 배포

### 1. GitHub 리포지토리 준비

필요한 파일:
```
your-repo/
├── streamlit_app.py
├── requirements.txt
├── chroma_db_store/          # Chroma 벡터 DB
│   └── ...
└── .streamlit/
    └── secrets.toml          # (로컬 테스트용, .gitignore에 추가)
```

### 2. Streamlit Share 설정

1. [share.streamlit.io](https://share.streamlit.io) 접속
2. "New app" 클릭
3. GitHub 리포지토리 연결
4. **Settings > Secrets**에 추가:
   ```toml
   OPENAI_API_KEY = "sk-your-actual-api-key"
   ```

### 3. 주의사항

- `chroma_db_store/` 폴더가 반드시 포함되어야 함
- Streamlit Share 무료 플랜은 리소스 제한이 있음
- 대용량 벡터 DB의 경우 외부 호스팅 고려 (Pinecone, Weaviate 등)

## 파일 구조

```
├── streamlit_app.py          # Streamlit 앱 메인
├── langgraph_rag_chatbot_v3.py  # CLI 버전 (참고용)
├── requirements.txt          # 의존성
├── chroma_db_store/          # 벡터 DB
└── .streamlit/
    └── secrets.toml.example  # secrets 예시
```

## 질문 예시

- "2024년 청소년 과의존률은?"
- "2021년부터 2024년까지 학령별 과의존률 변화"
- "숏폼 이용과 과의존의 관계는?"
- "가구원 수에 따른 과의존률 비교"

## 기술 스택

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o / GPT-4o-mini
- **Embedding**: text-embedding-3-large
- **Vector DB**: Chroma
- **Orchestration**: LangGraph
