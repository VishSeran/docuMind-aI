from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#define llm model
def llm_model ():
    
    try:
        model_name = "llama3.1"
        
        llm = OllamaLLM(
            model=model_name,
            temperature=0.5,
        )
        
        return llm
    
    except Exception as e:
        print(f"Unknown error: {e}")
        return None

#define embedding model
def embeddings():
    
    try:
        embed_model = HuggingFaceEmbeddings(
            model = "BAAI/bge-m3"
        )
        
        return embed_model
    
    except Exception as e:
        print(f"Unknown error: {e}")
        return None

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
        return None
    except Exception as e:
        print(f"Unknown error: {e}")
        return None

#define text splitter
def text_splitters(data, chunk_size, chunk_overlap):
    
    try:
        
        if data is None:
            raise ValueError("file is none")
        
        if isinstance(data, list) and len(data) ==0:
            raise ValueError ("file is empty")
        
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size = chunk_size,
            chunk_overlap=chunk_overlap,
            length_function = len
        )
        
        chunks = splitter.split_documents(data)
        return chunks
    
    except ValueError as e:
        print(f"value error:{e}")
        return None
    
    except Exception as e:
        print(f"Unknown error: {e}")
        return None
    
    
    