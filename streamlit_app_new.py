import streamlit as st
import tiktoken
from loguru import logger
import os
import tempfile

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
    st.set_page_config(
        page_title="Library and Information Science OpenChat",
        page_icon=":books:"
    )

    st.title(" :red[:books: LISTBOT] 에게 물어보세요 :grey_exclamation:")
    st.caption(" :closed_book: Hello, welcome to the Library and Information Science Q&A chat. ")
    st.caption(" :closed_book: This app is developed for the Introduction to Data Science course project for Spring 2024 at Chung-ang University. Feel free to ask any questions to LISBOT. Whether you're looking for research help, resource recommendations, or answers to specific questions, LISBOT is here to assist you.")
    
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
        if not openai_api_key:
            st.info("OpenAI API key를 다시 입력하세요.")
            st.stop()

        files_text = get_text(uploaded_files)
        if not files_text:
            st.error("파일에서 텍스트를 추출하지 못했습니다.")
            return

        text_chunks = get_text_chunks(files_text)
        if not text_chunks:
            st.error("텍스트 청크를 생성하지 못했습니다.")
            return
        
        st.write(f"text_chunks 길이: {len(text_chunks)}")
        
        vectorstore = get_vectorstore(text_chunks)
        if not vectorstore:
            st.error("벡터 스토어를 생성하지 못했습니다.")
            return

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요 저는 문헌정보학 전문 AI 사서 LISBOT입니다. 첨부한 자료를 기반으로 답변을 제공해 드립니다. 자료를 업로드하고 궁금한 점을 저에게 물어보세요 :speech_balloon:"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("여기에 질문을 입력하세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    
    for doc in docs:
        file_name = doc.name
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(doc.getvalue())
            temp_file_path = temp_file.name
            logger.info(f"Uploaded {file_name}")

        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        elif file_name.endswith('.docx'):
            loader = Docx2txtLoader(temp_file_path)
        elif file_name.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(temp_file_path)
        
        documents = loader.load_and_split()
        doc_list.extend(documents)
        os.remove(temp_file_path)  # 임시 파일 삭제

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
    
    # 디버깅: text_chunks 출력
    st.write(f"text_chunks: {text_chunks}")

    embeddings_list = embeddings.embed_documents([chunk.page_content for chunk in text_chunks])
    st.write(f"embeddings 길이: {len(embeddings_list)}")

    if not embeddings_list:
        return None

    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore

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

if __name__ == '__main__':
    main()

