import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import  google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_docs(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    api_key = ''  # Replace 'YOUR_API_KEY' with your actual Google API key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")
    # return vector_store
    
def get_conversational_chain():
    prompt_template="""
    Answer the following questions as concise as you can from the provided context, make sure to provide all the details, if the answer is not in the provided context, don't provide the wrong answer and just say DON'T KNOW.
    \n\n
    Context:\n{context}\n
    Query:\n{query}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context','query'])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    chain = get_conversational_chain()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index",embeddings)
    docs = db.similarity_search(user_question)
    response = chain({'input_documents':docs,'query':user_question},return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])
    

def main():
    st.set_page_config("Chat With Multi PDF")
    st.header("Chat With PDF")
    
    user_question = st.text_input("Ask a Question:")
    
    if user_question:
        user_input(user_question)
        
    with st.sidebar:
        st.title("Menu: ")
        pdf_docs = st.file_uploader("Upload Your PDF Files and Click on the Submit Button", accept_multiple_files=True)
        print("**********************")
        print(type(pdf_docs))
        print("***********************")
        if st.button("Submit & Process"):
            with st.spinner("Processing....."):
                raw_text = get_pdf_docs(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vectorstore(text_chunks=text_chunks)
                st.success("Done")
                
if __name__ == "__main__":
    main()