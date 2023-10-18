import os
import sys
import openai
from environs import Env
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import UnstructuredXMLLoader
from langchain.text_splitter import CharacterTextSplitter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from settings import DATA_DIR_PATH, ENV_DIR_PATH

env = Env()
env.read_env(os.path.join(ENV_DIR_PATH, '.env'))
openai.api_key = env.str("HUGGINGFACE_API_TOKEN")

def setup_huggingface_pipeline(model_id, task, pipeline_kwargs):
    hf_pipeline = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task=task,
        pipeline_kwargs=pipeline_kwargs,
    )
    return hf_pipeline

def run_llm(query: str):
    data_path = os.path.join(DATA_DIR_PATH, 'test_data.xml')
    loader = UnstructuredXMLLoader(data_path, mode="elements", strategy="fast",)
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents=docs)

    model_id = "gpt2"
    task = "text-generation"
    pipeline_kwargs = {"max_new_tokens": 10}
    hf_pipeline = setup_huggingface_pipeline(model_id, task, pipeline_kwargs)

    result = hf_pipeline(query)
    return result


if __name__ == "__main__":
    while True:
        print(run_llm(input("Enter question: ")))