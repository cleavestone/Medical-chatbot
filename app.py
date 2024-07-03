from flask import Flask,render_template,jsonify,request
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
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

app=Flask(__name__)
load_dotenv()

llm=OpenAI()
embeddings=OpenAIEmbeddings()

api_key=os.getenv("OPENAI_API_KEY")
pinecone_key=os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"]=api_key
os.environ["PINECONE_API_KEY"]=pinecone_key

index_name="chatbot2"
pc = Pinecone()

embeddings=OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
docsearch=vectorstore.from_existing_index(index_name=index_name,embedding=embeddings)

PROMPT=PromptTemplate(template=prompt_template,input_variables=["context","question"])
llm=OpenAI()

qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=docsearch.as_retriever(),
    return_source_documents=True
)

# create default route
@app.route('/')
def index():
    return render_template('chat.html')




