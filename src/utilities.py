from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
#define llm model
def llm_model (model_id):
    model_name = "llama3.1"
    
    llm = OllamaLLM(
        model=model_name,
        temperature=0.5,
    )
    
    return llm

#define embedding model
def embeddings():
    
    embed_model = HuggingFaceEmbeddings(
        model = "BAAI/bge-m3"
    )
    
    return embed_model


    
    
    
    