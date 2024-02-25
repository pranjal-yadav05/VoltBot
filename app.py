# Add your Google GenerativeAI API KEY as a system environment variable, name the variable as "GOOGLE_API_KEY"


from flask import Flask, render_template, request, redirect, url_for 
import os
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from PyPDF2 import PdfReader
import  google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

app = Flask(__name__)

def update_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists("faiss_index"):
        existing_vector_store = FAISS.load_local("faiss_index", embeddings)
        existing_vector_store.add_texts(text_chunks)
        existing_vector_store.save_local("faiss_index")
    else:
        print("No existing vector store found. Please upload a PDF to create the index.")

# Function to configure GenAI
def configure_genai():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get text from PDF
def get_pdf_text(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Function to create vector store
def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to load vector store
def load_vector_store(embeddings):
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings)
    else:
        return None

# Function to get conversational chain
def get_conversational_chain():
    configure_genai()                                   # Configure GenAI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    prompt_template = """
    You are a Friendly AI assistant, Expert in Power Substation Maintenance. 
    You should answer questions related to Power Substation Maintenance, otherwise reply sarcastically.
    If the question asked is of the field of Electrical engineering, you can provide suitable answer yourself even if context is not provided.
    Assist users by generating responses to their inquiries based on the provided context.
    Answer questions in a structured format.
    Deliver: detailed and easy-to-understand answers.
    Neutrality: Avoid expressing personal opinions or biases.
    Relevance: Focus on providing information directly related to the user's query.
    \n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to get user input and generate response
def get_user_input(user_question, vector_store, conversational_chain):
    print("User Question:", user_question)                  # Add this debug statement
    docs = vector_store.similarity_search(user_question)
    print("Docs:", docs)                                    # Add this debug statement
    response = conversational_chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print("Response:", response)                            # Add this debug statement
    return response["output_text"]


# Main route
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/redirect_to_chat')
def redirect_to_chat():
    return redirect(url_for('chat'))                        # Redirect to the chat route

@app.route('/redirect_to_index')
def redirect_to_index():
    return redirect(url_for('index')) 

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    if request.method == 'POST':
        # Initialize chat components
        vector_store = load_vector_store(embeddings)
        conversational_chain = get_conversational_chain()

        user_question = request.form.get('question')        # Use get method to handle missing key gracefully

        if user_question:
        # Get user input and generate response
            response = get_user_input(user_question, vector_store, conversational_chain)
            return render_template('chat.html', question=user_question, response=response)

    # If no file was uploaded or no question was provided, simply load the chat page
    return render_template('chat.html', question=None, response=None)


@app.route('/upload', methods=['POST'])
def upload():
    if 'pdfFile' not in request.files:
        return redirect(request.url)
    file = request.files['pdfFile']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Read PDF file content and extract text
        text = get_pdf_text(file)
        # Split text into chunks
        text_chunks = get_text_chunks(text)
        # Update vector store with new text chunks
        update_vector_store(text_chunks)
        return redirect(url_for('chat'))        # Redirect to chat page
    return redirect(url_for('index'))           # Redirect to index if something goes wrong

def record_audio():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

if __name__ == "__main__":
    app.run(debug=True)
