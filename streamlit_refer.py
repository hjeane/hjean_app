import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    # st.set_page_config(
    #     page_title="Library and Information Science OpenChat",
    #     page_icon=":books:"
    # )

    st.title("📚 LISBOT에게 물어보세요 ❕")
    st.caption("📢 *Welcome to the Library and Information Science Q&A chat. This app is developed for the Introduction to Data Science course project for Spring 2024. Feel free to ask any questions to LISBOT. Whether you're looking for research help, resource recommendations, or answers to specific questions, LISBOT is here to assist you.*")
    st.caption("✔️ **반드시 파일을 먼저 첨부한 뒤 OPENAPI KEY를 입력해주세요** 🖋️")
    st.caption("✔️ **LISBOT은 첨부한 자료를 기반으로 한 답변을 제공합니다. 자료를 업로드하고 궁금한 점을 물어보세요** 💬")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("여기에 파일을 업로드하세요.", type=['pdf', 'docx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        if not uploaded_files:
            st.info("먼저 파일을 업로드하세요.")
            st.stop()
        if not openai_api_key:
            st.info("OpenAI API key를 다시 입력하세요.")
            st.stop()

        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요 저는 ai 사서 LISBOT입니다. 궁금한 것이 있나요?"}]

    # CSS for chat messages
    st.markdown("""
        <style>
            .chat-container {
                display: flex;
                flex-direction: column;
                max-width: 600px;
                margin: 0 auto;
            }
            .chat-message {
                display: flex;
                align-items: flex-start;
                margin: 10px 0;
                padding: 10px;
                border-radius: 10px;
                max-width: 80%;
            }
            .chat-message.user {
                background-color: #e6ffe6;
                align-self: flex-end;
            }
            .chat-message.assistant {
                background-color: #f0f2f6;
                align-self: flex-start;
            }
            .chat-message .avatar {
                width: 40px;
                height: 40px;
                margin-right: 10px;
            }
            .chat-input-container {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 10px;
                background-color: #f0f2f6;
                border-top: 1px solid #ccc;
            }
            .chat-input {
                width: 100%;
                padding: 10px;
                border-radius: 20px;
                border: 1px solid #ccc;
                margin-right: 10px;
                outline: none;
            }
            .chat-button {
                padding: 10px 20px;
                border: none;
                border-radius: 20px;
                background-color: #4CAF50;
                color: white;
                cursor: pointer;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown(f"""
                <div class='chat-message assistant'>
                    <span class='avatar'>🤖</span>
                    <span>{message['content']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='chat-message user'>
                    <span class='avatar'>🧑</span>
                    <span>{message['content']}</span>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    query = st.text_input("여기에 질문을 입력하세요.", key="chat_input")
    if st.button("Send", key="send_button"):
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            st.markdown(f"""
                <div class='chat-message user'>
                    <span class='avatar'>🧑</span>
                    <span>{query}</span>
                </div>
                """, unsafe_allow_html=True)

            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(f"""
                    <div class='chat-message assistant'>
                        <span class='avatar'>🤖</span>
                        <span>{response}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help=source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help=source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help=source_documents[2].page_content)

            # Add assistant message to chat history
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

def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return

