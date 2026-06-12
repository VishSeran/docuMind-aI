from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
#define llm model
def llm_model (model_id):
    
    try:
        model_name = "llama3.1"
        
        llm = OllamaLLM(
            model=model_name,
            temperature=0.5,
        )
        
        return llm
    
    except Exception as e:
        print(f"Unknow error: {e}")

#define embedding model
def embeddings():
    
    try:
        embed_model = HuggingFaceEmbeddings(
            model = "BAAI/bge-m3"
        )
        
        return embed_model
    
    except Exception as e:
        print(f"Unknow error: {e}")

#define PDF loader
def pdf_loader(file_path):
    
    try:
        if not file_path:
            raise ValueError("file path is empty or none")
        
        pfdLoader = PyPDFLoader(file_path=file_path)
        PdfData = pfdLoader.load()
        return PdfData
    
    except ValueError as e:
        print(f"value error: {e}")  
    except Exception as e:
        print(f"Unknow error: {e}")
    
    