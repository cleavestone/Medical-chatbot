
from langchain.document_loaders import PyMuPDFLoader,PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(directory):
    loader=PyPDFDirectoryLoader(directory,glob='*.pdf')
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    documents=text_splitter.split_documents(data)
    return documents