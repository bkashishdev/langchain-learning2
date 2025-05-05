import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
# from langchain_community.embeddings import OllamaEmbeddings

from langchain_ollama import OllamaEmbeddings


from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import uuid

# Initialize session state
if 'store' not in st.session_state:
    st.session_state.store = {}

load_dotenv()


st.title("Conversational RAG With PDF Upload ")
st.write("Upload PDF and chat with their content")

# Use user-provided API key
api_key = st.text_input("Enter your Groq API Key:", type="password")

if api_key:
    # Initialize Groq with user-provided key
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=api_key)
    
    # Generate unique session ID if not provided
    session_id = st.text_input("Session ID", value=str(uuid.uuid4()))

    # File upload handling
    uploaded_files = st.file_uploader("Choose PDF Files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            # Create unique temp file path
            temp_path = f"./temp_{uuid.uuid4()}.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                documents.extend(docs)
            finally:
                # Clean up temp file
                os.remove(temp_path)

        # Document processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=OllamaEmbeddings(model="gemma:2b")
        )
        retriever = vectorstore.as_retriever()

        # RAG chain setup
        contextualize_q_system_prompt = """Given a chat history and the latest user question,
        which might reference context in the chat history, formulate a standalone question
        which can be understood without the chat history."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_system_prompt = """You are an assistant for question-answering tasks.
        Use the retrieved context to answer the question. If you don't know the answer,
        say so. Keep answers concise (3 sentences max).\n\n{context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Session history management
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Chat interface
        user_input = st.text_input("Your question:")
        if user_input:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            # Display response
            st.success(f"Assistant: {response['answer']}")
            
            # Display chat history
            session_history = get_session_history(session_id)
            st.write("### Chat History")
            for msg in session_history.messages:
                st.write(f"{msg.type.capitalize()}: {msg.content}")

else:
    st.warning("Please enter your Groq API key to continue")