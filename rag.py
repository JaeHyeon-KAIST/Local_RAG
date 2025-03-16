# 필요한 라이브러리 임포트
import streamlit as st  # 웹 애플리케이션 구축을 위한 라이브러리
from langchain_community.document_loaders import PyPDFLoader  # PDF 파일을 로드하고 텍스트를 추출하는 도구
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 긴 텍스트를 작은 청크로 분할하는 도구
from langchain_community.embeddings import OllamaEmbeddings  # Ollama를 사용한 텍스트 임베딩 생성
from langchain_community.vectorstores import Chroma  # 벡터 데이터베이스 저장소
from langchain_community.llms import Ollama  # Ollama LLM(대규모 언어 모델) 인터페이스
from langchain.chains import RetrievalQA  # 질의응답 체인 구현
import tempfile  # 임시 파일 처리
import os  # 파일 시스템 작업
import time  # 시간 측정을 위한 모듈 추가
import subprocess  # 터미널 명령 실행

# Streamlit 페이지 기본 설정
st.set_page_config(
  page_title="AI 논문 분석기",
  page_icon="📚",
  layout="wide"
)

# 메인 페이지 제목
st.title("📚 AI 논문 분석 도우미 with Ollama")


# 1️⃣ 로컬에 설치된 Ollama 모델 목록 가져오기
def get_ollama_models():
  try:
    # result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    # ollama list는 설치 순서대로 출력
    result = subprocess.run("ollama list | sort -k1", capture_output=True, text=True, shell=True)
    models = [line.split()[0] for line in result.stdout.splitlines() if line]
    return models[1:]
  except Exception as e:
    st.error(f"Ollama 모델 목록을 불러오는 데 실패했습니다: {e}")
    return []


# 사용자가 보유한 Ollama 모델 리스트 불러오기
available_models = get_ollama_models()

# 모델 선택 UI (기본값: deepseek-r1:14b)
selected_model = st.sidebar.selectbox("사용할 Ollama 모델을 선택하세요:", available_models, index=available_models.index(
  "deepseek-r1:14b") if "deepseek-r1:14b" in available_models else 0)

# PDF 파일 업로드 위젯
uploaded_file = st.file_uploader("논문 PDF 파일을 업로드하세요", type=['pdf'])

# 2️⃣ 파일이 업로드되면 실행되는 로직
if uploaded_file is not None:
  # 업로드된 파일을 임시 파일로 저장
  with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
    tmp_file.write(uploaded_file.getvalue())
    temp_path = tmp_file.name  # 임시 파일 경로 저장

  try:
    # PDF 파일 로드 및 텍스트 추출
    loader = PyPDFLoader(temp_path)
    pages = loader.load()

    # 텍스트를 작은 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)

    # 3️⃣ 선택한 모델을 기반으로 텍스트 임베딩 설정
    embeddings = OllamaEmbeddings(
      model="nomic-embed-text",
      base_url="http://localhost:11434"
    )

    # 벡터 저장소 설정 및 문서 저장
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./.chroma")

    # 4️⃣ 선택한 모델을 기반으로 LLM 설정
    llm = Ollama(
      model=selected_model,  # 사용자가 선택한 모델 반영
      temperature=0,
      base_url="http://localhost:11434"
    )

    # 질의응답 체인 생성
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    # 사용자 질문 입력 받기
    user_question = st.text_input("논문에 대해 질문해보세요:")

    # 질문이 입력되면 실행
    if user_question:
      start_time = time.time()

      with st.spinner('답변을 생성하고 있습니다...'):
        progress_placeholder = st.empty()
        progress_placeholder.text("답변을 생성 중입니다...")

        # 질문에 대한 답변 생성
        response = qa_chain.invoke({"query": user_question})

        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)

        progress_placeholder.empty()

        # 답변 출력
        st.write("### 🤖 AI 답변:")
        st.write(response['result'])
        st.info(f"⏱️ 답변 생성 소요 시간: {elapsed_time}초")

  finally:
    # 임시 파일 삭제
    os.unlink(temp_path)

# 사이드바에 사용 설명 추가
with st.sidebar:
  st.header("사용 방법")
  st.write("""
    1. 사용할 Ollama 모델을 선택하세요.
    2. PDF 논문 파일을 업로드하세요.
    3. 논문에 대해 궁금한 점을 질문하세요.
    4. AI가 논문을 분석하여 답변을 제공합니다.
    """)
