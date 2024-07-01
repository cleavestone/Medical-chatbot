from src.helper import load_pdf
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
import pinecone
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
pinecone_key=os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"]=api_key
os.environ["PINECONE_API_KEY"]=pinecone_key

## Connect to Pinecone server
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

pc = Pinecone()

index_name = "chatbot2"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 

embeddings=OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)


extracted_data=load_pdf(r'C:\Users\Hp\Desktop\Medical-chatbot-llama2\data')

vectorstore_from_docs = PineconeVectorStore.from_documents(
        extracted_data,
        index_name=index_name,
        embedding=embeddings
)