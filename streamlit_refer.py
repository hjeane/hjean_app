import streamlit as st
import tiktoken
from loguru import logger
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def main():
    st.set_page_config(
        page_title="Library and Information Science OpenChat",
        page_icon=":books:")

    st.title(" :red[ :books: LISBOT] 에게 물어보세요	:grey_exclamation:")
    st.caption("        :loudspeaker: *Welcome to the Library and Information Science Q&A chat. This app is developed for the Introduction to Data Science course project for Spring 2024.Feel free to ask any questions to LISBOT. Whether you're looking for research help, resource recommendations, or answers to specific questions, LISBOT is here to assist you.*")
    st.caption("    	:heavy_check_mark: 	**LISBOT은 첨부한 자료를 기반으로 한 답변을 제공합니다. 자료를 업로드하고 궁금한 점을 물어보세요**     :speech_balloon:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("여기에 파일을 업로드하세요.", type=['pdf', 'docx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        data4library_api_key = st.text_input("Data4Library API Key", key="data4library_api_key", type="password")
        process = st.button("Process")

    if process:
        if not uploaded_files:
            st.info("먼저 파일을 업로드하세요.")
            st.stop()
        if not openai_api_key:
            st.info("OpenAI API key를 다시 입력하세요.")
            st.stop()

        try:
            files_text = get_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
            st.session_state.processComplete = True
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error processing files: {str(e)}")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요 저는 ai 사서 LISBOT입니다. 궁금한 것이 있나요?"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # 채팅 로직
    if query := st.chat_input("여기에 질문을 입력하세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            if not chain:
                st.error("Process is not complete. Please upload files and process them first.")
                st.stop()

            with st.spinner("Thinking..."):
                try:
                    result = chain({"question": query})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    source_documents = result['source_documents']

                    # Data4Library API 호출
                    data4library_api_key = st.session_state.get("data4library_api_key")
                    if data4library_api_key:
                        data4library_response = get_data4library_response(data4library_api_key, query)
                        if data4library_response:
                            response += "\n\n**Data4Library API Response:**\n"
                            response += str(data4library_response)  # 필요에 따라 응답 형식 조정

                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:
                            st.markdown(doc.metadata['source'], help=doc.page_content)

                except Exception as e:
                    st.error(f"An error occurred during the chat: {str(e)}")
                    logger.error(f"Chat error: {str(e)}")

        # 어시스턴트 메시지를 채팅 기록에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()
        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

def get_data4library_response(api_key, query):
    url = f"http://data4library.kr/api/srchApiData?authKey={api_key}&query={query}&output=json"
    
    # Retry 설정 추가
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    http = requests.Session()
    http.mount("http://", adapter)
    http.mount("https://", adapter)

    try:
        response = http.get(url, timeout=30)  # 타임아웃 설정 (30초)
        response.raise_for_status()  # HTTP 오류가 발생했는지 확인
        return response.json()
    except requests.exceptions.Timeout:
        logger.error("The request timed out")
        return {"error": "The request timed out"}
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    main()
