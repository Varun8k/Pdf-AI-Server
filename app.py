from flask import Flask, request, jsonify
import pdfplumber
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

faiss_index = None
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

def get_pdf_text(pdf_file):
    """
    Extract text from PDF documents using pdfplumber.
    """
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, chunk_size=10000, chunk_overlap=1000):
    """
    Split text into manageable chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def get_vector_store(text_chunks):
    global faiss_index
    """
    Generate vector store from text chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index = FAISS.from_texts(text_chunks, embedding=embeddings)
   

def get_conversational_chain():
    """
    Create conversational chain for question answering.
    """
    prompt_template = """
   Generate a detailed and accurate answer based on the provided context and question. Ensure that the response covers all relevant aspects of the topic. If the answer is not readily available in the provided context, please indicate "answer not available in pdf," but make an attempt to provide relevant insights if possible. Avoid speculation and refrain from providing incorrect information. Craft a thorough and thoughtful response that adds value to the user's query. If you don't have data about the question, you can provide basic information such as definitions, explanations, and general insights related to the topic. If still no relevant data is found, simply say "sorry no latest relevant data found".
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 0,
        "max_output_tokens": 8192,
    })

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    global faiss_index
    """
    Handle user input for question answering.
    """
    new_db =faiss_index
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

@app.route('/')
def index():
    return jsonify({'message': "Hlo Server"})

@app.route('/reset')
def reset():
    global faiss_index
    faiss_index = None
    return jsonify({'message': "FAISS index reset"})

@app.route('/pdf', methods=['POST'])
def process():
    pdf_file = request.files['pdf_file']
    raw_text = get_pdf_text(pdf_file)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return jsonify({'response': 'successful'})

@app.route('/question', methods=['POST'])
def handle_question():
    request_data = request.json 
    print("Request Data:", request_data) 
    user_question = request_data.get('question') 
    if user_question is None:
        return jsonify({'error': 'Question field is missing'})
    response = user_input(user_question)
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')