import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv
load_dotenv()
# ## load the GROQ API Key
# os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
# groq_api_key=os.getenv("GROQ_API_KEY")

# Set page title and favicon (Ananda logo)
logo_path = "https://ananda.exchange/wp-content/uploads/2022/03/cropped-Fondos-y-recursos-20.png"  # Replace with the path to your Ananda logo
st.set_page_config(
    page_title="ChatBot",
    page_icon=logo_path  # Setting the favicon (logo) in the tab
)

# Access the secrets via st.secrets
groq_api_key = st.secrets["GROQ_API_KEY"]
hf_token = st.secrets["HF_TOKEN"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
langchain_project = st.secrets["LANGCHAIN_PROJECT"]


# ## Langsmith Tracking
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

## If you do not have open AI key use the below Huggingface embedding
#os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context and also Imagine yourself an an expert in the field of cryptocurrency.
    Please provide the most accurate response based on the question.
    If you don't know the answer, try to give generic answer
    <context>
    {context}
    <context>
    Question:{input}

    """

)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFDirectoryLoader("Docs") ## Data Ingestion step
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

# Inject CSS with base64 background image
st.markdown(
    """
    <style>
    .chat-container {
        max-width: 700px;
        margin: 0 auto;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }

    # .chatbox {
    #     border-radius: 15px;
    #     padding: 10px;
    #     margin-bottom: 10px;
    #     max-width: 60%;
    #     display: inline-block;
    #     word-wrap: break-word;
    # }

    .user-msg {
        background-color: #A9A9A9;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        max-width: 45%;
        margin-right: auto;
        float: left;
        clear: both;
        word-wrap: break-word; /* Prevents text overflow */
    }

    .bot-msg {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        max-width: 45%;
        margin-left: auto;
        float: right;
        clear: both;
        word-wrap: break-word; /* Prevents text overflow */
    }

    .input-area {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }

    .input-box {
        width: 80%;
        padding: 10px;
        border-radius: 20px;
        border: 1px solid #ccc;
        background-color: black; /* Change input background to black */
        color: white;
    }

    .send-button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 50%;
        padding: 12px;
        margin-left: 10px;
        cursor: pointer;
    }

    /* Scrollable chat container */
    .scrollable-chat {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 15px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Display the logo at the top left corner using st.image
logo_path = "https://ananda.exchange/wp-content/uploads/2022/03/cropped-Fondos-y-recursos-20.png"
st.image(logo_path, width=100)  # Adjust the width if needed

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main UI components
st.title("Master Crypto with ANANDA")

user_prompt=st.text_input("Message Ananda")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("I'm ready")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    response_text = response['answer']

# Append the new user message and the response to chat history
    st.session_state.chat_history.append({"user": user_prompt, "bot": response_text})

    #newline1 starts
    # Chat Container
    st.markdown('<div class="chat-container scrollable-chat">', unsafe_allow_html=True)
    #newline1 ends

    # Display the entire chat history
    for chat in reversed(st.session_state.chat_history):   
        #new line2 starts 
        user_msg = f'<div class="user-msg">You: {chat["user"]}</div>'
        bot_msg = f'<div class="bot-msg">Ananda: {chat["bot"]}</div>'
        st.markdown(user_msg + bot_msg, unsafe_allow_html=True)
     
    st.markdown('</div>', unsafe_allow_html=True)
    
    print(f"Response time: {time.process_time() - start}")

    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
