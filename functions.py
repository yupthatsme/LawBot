from langchain.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader, DataFrameLoader
import tempfile
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Pinecone
import uuid
import pinecone
from dotenv import load_dotenv
import pandas as pd


def initialize_pinecone():
  
    load_dotenv()  # Load environment variables from .env file

    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENV")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key or not environment:
        raise ValueError("Pinecone API key or environment not set in .env file")

   # Initialize Pinecone without arguments
    pc = Pinecone()

    # Set the API key and environment using the provided methods
    pc.init(api_key=api_key, environment=environment)
    # Check if the index exists, if not, create it
    if index_name not in pc.list_indexes().names():
        # Create the index (adjust 'dimension' and 'metric' according to your needs)
        pc.create_index(
            name=laws,
            dimension=1536,  # Adjust this to match the dimension of your embeddings
            metric='cosine',  # Can be 'cosine', 'dotproduct', etc.
            spec=ServerlessSpec(
                cloud='aws',   # Your cloud provider
                region=us-east-1  # Your Pinecone environment (e.g., 'us-west-1')
            )
        )

    return index_name
    
    
    
    # load_dotenv()

    # initialize pinecone
   # pinecone.init(
       # api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    #    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
   # )
   # index_name= os.getenv("PINECONE_INDEX_NAME")
   # if index_name not in pinecone.list_indexes():
        # Create an index if it doesn't exist, replace 1536 with the actual dimension of your embeddings
       # pinecone.create_index(index_name, dimension=1536)
   # return index_name

def save_to_pinecone(data, file_name):

    index_name= initialize_pinecone()

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()

    index = pinecone.Index(index_name)
    vectorstore = Pinecone(index, embeddings.embed_query, "text")
    vectorstore.add_documents(docs)



def save_to_disk(data, file_name):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()

    vector_db = FAISS.from_documents(
        documents=docs,
        embedding=embeddings,
    )
    folder_path = f'docs/{file_name}'
    
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    vector_db.save_local(folder_path)


def make_temp_file(file, file_name, extension):
    # Read the content of the uploaded file
    file_content = file.read()

    # Create a temporary file to store the content
    temp_file = tempfile.NamedTemporaryFile(delete=False, prefix=file_name, suffix=extension)
    temp_file.write(file_content)
    temp_file.close()
    return temp_file

def process_txt_file(file, file_name, file_extension):
    extension = f'.{file_extension}'
    temp_file = make_temp_file(file, file_name, extension)

    try:
        # Now you can pass the path of the temporary file to your loader
        loader = TextLoader(file_path={temp_file.name})
        data = loader.load()
        # print(data)
        save_to_pinecone(data, file_name)
    finally:
        # Clean up the temporary file
        os.remove(temp_file.name)

def process_pdf_file(file, file_name, file_extension):
    extension = f'.{file_extension}'
    temp_file = make_temp_file(file, file_name, extension)

    try:
        # Now you can pass the path of the temporary file to your loader
        loader = PyPDFLoader(file_path=temp_file.name)
        data = loader.load()
        # print(len(data))
        save_to_pinecone(data, file_name)
    finally:
        # Clean up the temporary file
        os.remove(temp_file.name)

def process_docx_file(file, file_name, file_extension):
    extension = f'.{file_extension}'
    temp_file = make_temp_file(file, file_name, extension)

    try:
        # Now you can pass the path of the temporary file to your loader
        loader = Docx2txtLoader(file_path=temp_file.name)
        data = loader.load()
        # print(len(data))
        save_to_pinecone(data, file_name)
    finally:
        # Clean up the temporary file
        os.remove(temp_file.name)

def process_spreadsheet_file(file, file_name, file_extension):
    extension = f'.{file_extension}'
    temp_file = make_temp_file(file, file_name, extension)

    try:
        # Now you can pass the path of the temporary file to your loader
        loader = UnstructuredExcelLoader(file_path=temp_file.name)
        data = loader.load()
        save_to_pinecone(data, file_name)
    finally:
        # Clean up the temporary file
        os.remove(temp_file.name)

def process_csv_file(file, file_name, file_extension):
    extension = f'.{file_extension}'
    temp_file = make_temp_file(file, file_name, extension)

    try:
        # Now you can pass the path of the temporary file to your loader
        loader = CSVLoader(file_path=temp_file.name, encoding='utf-8', csv_args={
            'delimiter':','
        })
        data = loader.load()
        save_to_pinecone(data, file_name)
    finally:
        # Clean up the temporary file
        os.remove(temp_file.name)
