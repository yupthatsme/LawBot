from langchain.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader, DataFrameLoader
import tempfile
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Pinecone
import uuid
from pinecone import Pinecone, Index, ServerlessSpec
from dotenv import load_dotenv
import pandas as pd

def initialize_pinecone():
    # Load Pinecone API key and environment variables
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")  # Use the actual Pinecone API key here
    environment = "us-east1-gcp"  # This is based on the provided region

    if not api_key:
        raise ValueError("Missing Pinecone API key in environment variables")

    # Initialize Pinecone with the correct environment
    pinecone.init(api_key=api_key, environment=environment)

    # Index information
    index_name = "quickstart"  # Correct index name based on your info

    # Check if the index exists, and create it if not
    if index_name not in pinecone.list_indexes():
        print(f"Creating new Pinecone index: {index_name}")
        pinecone.create_index(
            name=index_name,
            dimension=1536,  # Embedding size (adjust as necessary)
            metric="cosine"  # The similarity metric you are using
        )
    else:
        print(f"Pinecone index '{index_name}' already exists.")

    # Connect to the index
    index = pinecone.Index(index_name)
    return index

def save_to_pinecone(data, file_name):
    # Ensure the Pinecone index is initialized
    index = initialize_pinecone()

    # Split the document into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(data)

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Store the split documents into Pinecone vector store
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
        # Use the PyPDFLoader to load the PDF into a data structure
        loader = PyPDFLoader(file_path=temp_file.name)
        data = loader.load()
        
        # Store the data into Pinecone or another vector store
        save_to_pinecone(data, file_name)

    except Exception as e:
        print(f"Error processing PDF file: {e}")
    finally:
        # Clean up temporary file
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
