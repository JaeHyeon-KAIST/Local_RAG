import streamlit as st  # 웹 애플리케이션 구축을 위한 라이브러리
from langchain_community.document_loaders import PyPDFLoader  # PDF 파일을 로드하고 텍스트를 추출하는 도구
from langchain_docling import DoclingLoader  # 논문 로드를 위한 Docling 로더
from langchain_docling.loader import ExportType  # Docling 내보내기 타입
from langchain_text_splitters import MarkdownHeaderTextSplitter  # 마크다운 헤더 기반 텍스트 분할
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 긴 텍스트를 작은 청크로 분할하는 도구
from langchain.schema.runnable import Runnable
from langchain_community.vectorstores import Chroma  # 벡터 데이터베이스 저장소
import tempfile  # 임시 파일 처리
import os  # 파일 시스템 작업
import time  # 시간 측정을 위한 모듈 추가
import requests  # LM Studio API 호출을 위한 라이브러리
from chromadb.config import Settings

# Streamlit 페이지 기본 설정
st.set_page_config(
  page_title="AI 논문 분석기",
  page_icon="📚",
  layout="wide"
)

# 메인 페이지 제목
st.title("📚 AI 논문 분석 도우미 with LM Studio")

# LM Studio 서버 URL
LM_STUDIO_URL = "http://127.0.0.1:1234"


# LM Studio에서 사용 가능한 모델 목록 가져오기
def get_lm_studio_models():
  try:
    response = requests.get(f"{LM_STUDIO_URL}/v1/models")
    if response.status_code == 200:
      models_data = response.json()
      # API 응답 구조에 따라 모델 ID 또는 이름 추출
      if "data" in models_data:
        # OpenAI 호환 API 형식
        return [model["id"] for model in models_data["data"]]
      else:
        # 단순 목록 형식
        return models_data
    else:
      st.warning(f"LM Studio 모델 목록을 가져오는 데 실패했습니다. (상태 코드: {response.status_code})")
      return []
  except Exception as e:
    st.warning(f"LM Studio 연결 오류: {e}")
    return []


# LM Studio API를 통해 텍스트 임베딩 생성하는 클래스
class LMStudioEmbeddings:
  def __init__(self, api_url=f"{LM_STUDIO_URL}/v1/embeddings", model_name="text-embedding-nomic-embed-text-v1.5-embedding"):
    self.api_url = api_url
    self.model_name = model_name
    # 모델별 임베딩 차원 설정
    self.embedding_dim = 1024 if "bge" in model_name.lower() else 768

  def embed_documents(self, texts):
    embeddings = []
    for i, text in enumerate(texts):
      if i == 0:  # 첫 번째 문서의 임베딩 차원만 출력
        embedding = self.embed_query(text)
        embeddings.append(embedding)
      else:
        embeddings.append(self.embed_query(text))
    return embeddings

  def embed_query(self, text):
    headers = {
      "Content-Type": "application/json"
    }

    data = {
      "input": text,
      "model": self.model_name
    }

    response = requests.post(self.api_url, headers=headers, json=data)

    if response.status_code == 200:
      result = response.json()
      if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
        embedding = result["data"][0]["embedding"]
        actual_dim = len(embedding)
        if actual_dim != self.embedding_dim:
          st.warning(f"예상 임베딩 차원({self.embedding_dim})과 실제 임베딩 차원({actual_dim})이 다릅니다.")
          self.embedding_dim = actual_dim  # 실제 차원으로 업데이트
        return embedding
      else:
        # 임베딩이 제공되지 않는 경우, 가짜 임베딩 반환
        st.warning("LM Studio에서 임베딩을 제공하지 않았습니다. 대체 임베딩을 사용합니다.")
        return [0.0] * self.embedding_dim
    else:
      st.error(f"임베딩 API 요청 실패: {response.status_code}, {response.text}")
      return [0.0] * self.embedding_dim


# LM Studio API를 통해 추론하는 클래스 정의
class LMStudioLLM(Runnable):
  def __init__(self, api_url=f"{LM_STUDIO_URL}/v1/completions", model_name="deepseek-r1-distill-qwen-32b",
               temperature=0.1, max_tokens=2000, stop=None):
    self.api_url = api_url
    self.model_name = model_name
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.stop = stop if stop else []

  def invoke(self, *args, **kwargs):
    prompt = args[0] if args else kwargs.get('prompt')
    return self.run(prompt)

  def run(self, prompt):
    headers = {
      "Content-Type": "application/json"
    }

    data = {
      "prompt": prompt,
      "model": self.model_name,
      "temperature": self.temperature,
      "max_tokens": self.max_tokens,
      "stop": self.stop
    }

    response = requests.post(self.api_url, headers=headers, json=data)

    if response.status_code == 200:
      return response.json()["choices"][0]["text"]
    else:
      raise Exception(f"API 요청 실패: {response.status_code}, {response.text}")

  def __call__(self, prompt):  # __call__ 메서드 추가
    return self.run(prompt)


# LM Studio 서버 연결 확인
lm_studio_available = True
lm_studio_models = []

try:
  response = requests.get(f"{LM_STUDIO_URL}/v1/models")
  if response.status_code == 200:
    lm_studio_models = get_lm_studio_models()
  else:
    lm_studio_available = False
    st.warning("LM Studio 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
except Exception as e:
  lm_studio_available = False
  st.warning(f"LM Studio 서버 연결 오류: {e}")

# 사이드바 구성
st.sidebar.header("모델 설정")

# 추론 모델 선택 - LM Studio 사용
st.sidebar.subheader("추론(Reasoning) 모델 설정")
if lm_studio_available and lm_studio_models:
  reasoning_model = st.sidebar.selectbox(
    "LM Studio 추론 모델 선택:",
    lm_studio_models,
    index=lm_studio_models.index(
      "deepseek-r1-distill-qwen-32b") if "deepseek-r1-distill-qwen-32b" in lm_studio_models else 0,
    key="reasoning_model"
  )
else:
  st.sidebar.warning("LM Studio 연결 불가")
  reasoning_model = ""

# 답변 생성에 사용할 모델 선택 UI
st.sidebar.subheader("답변 생성 모델 설정")
if lm_studio_available and lm_studio_models:
  answer_model = st.sidebar.selectbox(
    "LM Studio 답변 생성 모델 선택:",
    lm_studio_models,
    index=lm_studio_models.index("exaone-3.5-7.8b-instruct") if "exaone-3.5-7.8b-instruct" in lm_studio_models else 0,
    key="answer_model"
  )
else:
  st.sidebar.warning("LM Studio 연결 불가")
  answer_model = ""

# 임베딩 모델 선택 UI 추가
embedding_model = st.sidebar.selectbox(
  "텍스트 임베딩에 사용할 인코더 모델을 선택하세요:",
  ["text-embedding-nomic-embed-text-v1.5-embedding", "text-embedding-bge-m3"],
  index=0,
  key="embedding_model"
)

# PDF 로더 선택 UI
loader_type = st.sidebar.radio(
  "PDF 로더 선택:",
  ["PyPDFLoader", "DoclingLoader"],
  help="PyPDFLoader는 일반적인 PDF 처리에 적합하고, DoclingLoader는 학술 논문의 구조를 더 잘 인식합니다."
)

# PDF 파일 업로드 위젯
uploaded_file = st.file_uploader("논문 PDF 파일을 업로드하세요", type=['pdf'])

vectorstore = None

# 2️⃣ 파일이 업로드되면 실행되는 로직
if uploaded_file is not None:
  # 업로드된 파일을 임시 파일로 저장
  with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
    tmp_file.write(uploaded_file.getvalue())
    temp_path = tmp_file.name  # 임시 파일 경로 저장

  try:
    # LM Studio 서버 연결 확인
    if not lm_studio_available:
      st.error("LM Studio 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
      st.stop()

    if loader_type == "PyPDFLoader":
      with st.spinner('PyPDFLoader로 논문을 로드 중...'):
        # PDF 파일 로드 및 텍스트 추출
        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        # 텍스트를 작은 청크로 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(pages)

    else:  # DoclingLoader
      with st.spinner('DoclingLoader로 논문을 로드 중...'):
        # Docling 로더를 사용하여 PDF 파일에서 논문 로드
        loader = DoclingLoader(
          file_path=temp_path,
          export_type=ExportType.MARKDOWN
        )
        docs = loader.load()

        # 마크다운 헤더 기반 텍스트 분할
        header_splitter = MarkdownHeaderTextSplitter(
          headers_to_split_on=[
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
          ]
        )

        # 각 문서를 헤더 기반으로 분할
        splits = []
        for doc in docs:
          try:
            header_splits = header_splitter.split_text(doc.page_content)
            splits.extend(header_splits)
          except Exception as e:
            st.warning(f"헤더 분할 중 오류 발생: {e}. 기본 분할 방식으로 전환합니다.")
            # 헤더 분할이 실패하면 기본 분할 방식 사용
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(doc.page_content)
            from langchain_core.documents import Document

            splits.extend([Document(page_content=chunk) for chunk in chunks])

    # Try to set up embeddings and vector store
    try:
      # 임베딩 설정
      embeddings = LMStudioEmbeddings(
        api_url=f"{LM_STUDIO_URL}/v1/embeddings",
        model_name=embedding_model
      )

      # 첫 번째 문서로 실제 임베딩 차원 확인
      if splits:
        first_embedding = embeddings.embed_query(splits[0].page_content)
        actual_dim = len(first_embedding)
      else:
        actual_dim = embeddings.embedding_dim

      # 모델 이름에 따라 벡터 저장소 디렉토리 설정
      persist_directory = f"./chroma{embedding_model}"

      # Chroma 설정 업데이트 (파일 시스템 기반)
      chroma_settings = Settings(
          anonymized_telemetry=False,
          is_persistent=True,  # 파일 시스템에 저장
          persist_directory=persist_directory  # 모델별 저장 경로 지정
      )

      # 새로운 벡터 저장소 생성
      vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_metadata={"dimension": actual_dim},
        client_settings=chroma_settings
      )

      if vectorstore is None:
        raise Exception("벡터 저장소 생성 실패")
      
    except Exception as e:
      st.error(f"임베딩 및 벡터 저장소 설정 오류: {e}")
      st.warning("벡터 검색 기능 없이 전체 문서를 사용합니다.")
      vectorstore = None

    # 4️⃣ LM Studio를 통한 추론 LLM 설정
    reasoning_llm = LMStudioLLM(
      api_url=f"{LM_STUDIO_URL}/v1/completions",
      model_name=reasoning_model,
      temperature=0,
      max_tokens=2000,
      stop=["</think>"]
    )

    # 답변 생성용 Ollama LLM 설정
    answer_llm = LMStudioLLM(
      api_url=f"{LM_STUDIO_URL}/v1/completions",
      model_name=answer_model,
      temperature=0,
      max_tokens=1000
    )


    # RetrievalQA 체인 클래스 정의
    class CustomRetrievalQA:
      def __init__(self, llm, retriever=None, documents=None):
        self.llm = llm
        self.retriever = retriever
        self.documents = documents

      def __call__(self, query_dict):
        query = query_dict["query"]

        if self.retriever:
          # 벡터 검색으로 관련 문서 가져오기
          docs = self.retriever.get_relevant_documents(query)
          context = "\n\n".join([doc.page_content for doc in docs])
        else:
          # 벡터 검색이 불가능한 경우 전체 문서 사용
          if self.documents:
            # 모든 문서의 내용을 결합하되 최대 길이 제한
            all_text = "\n\n".join([doc.page_content for doc in self.documents])
            # 토큰 길이 제한을 위해 텍스트 truncate (대략 10,000자 정도로)
            context = all_text[:10000] + ("..." if len(all_text) > 10000 else "")
            docs = self.documents[:5]  # 처음 5개 문서만 참조용으로 저장
          else:
            context = "문서 정보를 가져올 수 없습니다."
            docs = []

        prompt_text = f"""
            당신은 학술 논문을 분석하는 AI 어시스턴트입니다.
            다음은 사용자의 질문과 관련된 논문 내용입니다:

            {context}

            <think>
            위 정보를 바탕으로 다음 질문에 답변해주세요:
            {query}

            문서 정보를 기반으로 단계별로 추론해서 답변을 생성하세요.
            </think>
            """

        result = self.llm(prompt_text)

        return {
          "query": query,
          "result": result,
          "source_documents": docs
        }


    if vectorstore:
      retrieval_chain = CustomRetrievalQA(
        llm=reasoning_llm,
        retriever=vectorstore.as_retriever()
      )
    else:
      st.error("Vector store is not available. Please check the embeddings and vector store creation process.")
      st.stop()

    # 답변 생성을 위한 프롬프트 템플릿 정의
    answer_template = """
        당신은 학술 논문을 분석하고 질문에 답변하는 AI 어시스턴트입니다.

        다음은 질문에 대한 초기 분석 결과입니다:
        {reasoning_result}

        위 분석 결과를 바탕으로, 다음 질문에 대해 명확하고 간결하게 답변해주세요:
        {question}

        논문의 내용에 근거하여 답변하되, 학술적이고 정확한 표현을 사용하세요.
        """


    def generate_final_answer(llm, reasoning_result, question):
      prompt_text = answer_template.format(
        reasoning_result=reasoning_result,
        question=question
      )
      return llm(prompt_text)


    # 사용자 질문 입력 받기
    user_question = st.text_input("논문에 대해 질문해보세요:")

    # 질문이 입력되면 실행
    if user_question:
      start_time = time.time()

      with st.spinner('답변을 생성하고 있습니다...'):
        progress_placeholder = st.empty()
        progress_placeholder.text("1/2 단계: 관련 정보 검색 및 추론 중...")

        # 1단계: 추론 모델로 관련 정보 검색 및 분석
        reasoning_result = retrieval_chain({"query": user_question})
        reasoning_output = reasoning_result["result"]

        progress_placeholder.text("2/2 단계: 최종 답변 생성 중...")

        # 2단계: 답변 생성 모델로 최종 답변 만들기
        final_answer = generate_final_answer(
          answer_llm,
          reasoning_output,
          user_question
        )

        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)

        progress_placeholder.empty()

        # 답변 출력
        st.write("### 🤖 AI Answer:")
        st.write(final_answer)

        # 상세 정보 확인 옵션 (접이식)
        with st.expander("추론 과정 상세 보기"):
          st.markdown("#### 추론 모델의 분석 결과:")
          st.write(reasoning_output)

          st.markdown("#### 참조된 문서 출처:")
          for i, doc in enumerate(reasoning_result["source_documents"]):
            st.markdown(f"**문서 {i + 1}:**")
            st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
            st.markdown("---")

        st.info(f"⏱️ 답변 생성 소요 시간: {elapsed_time}초")

  finally:
    # 임시 파일 삭제
    os.unlink(temp_path)

# 사이드바에 사용 설명 추가
with st.sidebar:
  st.header("사용 방법")
  st.write("""
    1. 추론(Reasoning)에 사용될 모델을 선택하세요.
    2. 답변 생성에 사용할 모델을 선택하세요.
    3. PDF 논문 파일을 업로드하세요.
    4. 논문에 대해 궁금한 점을 질문하세요.
    5. AI가 두 단계로 논문을 분석하여 답변을 제공합니다:
       - 추론 모델이 관련 정보를 검색하고 분석합니다.
       - 답변 생성 모델이 분석 결과를 바탕으로 최종 답변을 생성합니다.
    """)

  st.markdown("---")

  st.header("두 단계 처리 과정")
  st.write("""
    **추론 단계 (Reasoning)**
    - 논문에서 질문과 관련된 정보를 검색합니다.
    - 검색된 정보를 바탕으로 초기 분석을 수행합니다.

    **답변 생성 단계 (Answer Generation)**
    - 추론 단계의 결과물을 바탕으로 더 정제된 최종 답변을 생성합니다.
    - 보다 자연스럽고 명확한 답변을 제공합니다.
    """)
