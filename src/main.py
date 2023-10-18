import os
import sys
import openai
from langchain.llms import Ollama
from langchain import FAISS
from environs import Env
from langchain.document_loaders import UnstructuredXMLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from settings import DATA_DIR_PATH, ENV_DIR_PATH

def run_llm(query: str):
    data_path = os.path.join(DATA_DIR_PATH, 'test_data.xml')
    loader = UnstructuredXMLLoader(data_path, mode="elements", strategy="fast",)
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents=docs)

    embeddings = OllamaEmbeddings(model="llama2:7b",)
    index_name = "med-chat"

    try:
        vectorstore = FAISS.load_local(index_name, embeddings)
    except Exception as e:
        print("Creating index...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(index_name)
        print("The index was created successfully")


    model = "llama2"
    ollama = Ollama(model=model)
    result = ollama(prompt=query)
    return result


if __name__ == "__main__":
    while True:
        print(run_llm(input("Enter question: ")))