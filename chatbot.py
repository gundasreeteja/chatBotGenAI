import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = ""#Provide your API key here

#Upload PDF files
st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)

    #Break the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    #Generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #Creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    #Get user question
    user_question = st.text_input("Type your question here")

    #Do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)

        #Define the LLM
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gtp-3.5-turbo"
        )

        #Output results
        #take the question, get relevant data, pass it to LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(
            input_documents=match,
            question=user_question
        )
        st.write(response)
