import os
import utils
import streamlit as st
from pathlib import Path
from streaming import StreamHandler
from functions import *
import shutil

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS, Pinecone
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="ChatDocs", page_icon="📄")

class CustomDataChatbot:

    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo-16k"

    def make_retriever(self):
        index_name= initialize_pinecone()
        embedding = OpenAIEmbeddings()
        docsearch = Pinecone.from_existing_index(index_name, embedding)
        return docsearch

    def save_file(self, file):
        # Get the file extension from the uploaded file's name
        file_extension = file.name.split('.')[-1].lower()
        file_name = file.name[:-(len(file_extension) + 1)]
        
        if file_extension == 'pdf':
            process_pdf_file(file, file_name, file_extension)
        elif file_extension == 'docx':
            process_docx_file(file, file_name, file_extension)
        elif file_extension in ['xls', 'xlsx']:
            process_spreadsheet_file(file, file_name, file_extension)
        elif file_extension == 'csv':
            process_csv_file(file, file_name, file_extension)
        elif file_extension == 'txt':
            process_txt_file(file, file_name, file_extension)
        else:
            print('File Format not supported')

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self):
        vectordb = self.make_retriever()

        llm = ChatOpenAI(temperature=0)

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            vectordb.as_retriever(),
            return_source_documents=True
        )
        return qa_chain
    
    def delete_documents(self):
        docs_folder = "docs"

        # List all subdirectories in the "docs" folder
        subdirectories = [subdir for subdir in os.listdir(docs_folder) if os.path.isdir(os.path.join(docs_folder, subdir))]

        for subdir in subdirectories:
            delete_button = st.sidebar.button(f"Delete {subdir}", key=subdir)
            if delete_button:
                dir_path = os.path.join(docs_folder, subdir)
                try:
                    shutil.rmtree(dir_path)  # Recursively remove directory and its contents
                    st.sidebar.success(f"Deleted Document: {subdir}")
                    st.experimental_rerun()  # Rerun the Streamlit app to update the buttons
                except Exception as e:
                    st.sidebar.error(f"Failed to delete document: {subdir}. Error: {e}")

    def get_sources(self, source_documents):
        metadata_strings = []
        for doc in source_documents:
            metadata = doc.metadata
            source_path = metadata['source']
            source_filename = os.path.basename(source_path)
            metadata_string = source_filename
            metadata_strings.append(metadata_string)

        # Join the metadata strings with '\n' separator
        # source_documents_string = "\n\n".join(metadata_strings)
        return 

    @utils.enable_chat_history
    def main(self):
        # tmp_files = os.listdir("docs")

        password = os.getenv('PASSWORD')

        user_password = st.sidebar.text_input('Enter Password')
        if user_password == password:
            st.sidebar.write('Make sure uploaded file name does not contain a "." example: file.name.pdf ')
            with st.sidebar.form("my-form", clear_on_submit=True):
                uploaded_files = st.file_uploader(label='Upload files', type=['pdf', 'docx', 'xls', 'xlsx', 'csv'], accept_multiple_files=True)
                submitted = st.form_submit_button("UPLOAD!")
            
                if submitted and uploaded_files is not None:
                    with st.spinner("Saving files..."):
                        for file in uploaded_files:
                            self.save_file(file)
                
        qa_chain = self.setup_qa_chain()


        
        user_query = st.chat_input(placeholder="Ask me anything!")
        chat_history = []
        
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = qa_chain({'question': user_query, 'chat_history': chat_history}, callbacks=[st_cb])

                source_documents_string = self.get_sources(response['source_documents'])

                st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                st.write(f'Source: {source_documents_string}')
                chat_history.append((user_query, response['answer']))


if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()