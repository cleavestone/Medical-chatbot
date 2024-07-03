from flask import Flask, render_template, jsonify, request
from langchain.vectorstores import Pinecone
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
from src.prompt import *
from langchain.embeddings import OpenAIEmbeddings
import os
from store_index import vectorstore_from_docs

app = Flask(__name__)
load_dotenv()

llm = OpenAI()
embeddings = OpenAIEmbeddings()

api_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
os.environ["PINECONE_API_KEY"] = pinecone_key

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_from_docs.as_retriever(),
    return_source_documents=True
)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get('msg', '')
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    
    result = qa({'query': msg})
    return jsonify({"result": result['result']})

if __name__ == "__main__":
    app.run(debug=True)


