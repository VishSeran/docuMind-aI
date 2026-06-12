from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever

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
def document_loader(file_path):
    
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
    

#define vector database
def vector_database(documents, embedding_model):
    
    try:
        if documents is None:
            raise ValueError("document is empty or None")
        
        chroma_db = Chroma.from_documents(documents=documents,
                                          embedding=embedding_model,
                                          persist_directory="chroma_db")
        
        return chroma_db
        
    except ValueError as e:
        print(f"value error: {e}")
        return None
    
    except Exception as e:
        print(f"Unknown error: {e}")
        return None
    
def retriever_call(file_path,parent_chunk_size=2000, 
            parent_chunk_overlap=400, child_chunk_size=500, child_chunk_overlap=50):
    
    try:
        
        if not file_path:
            raise ValueError ("Document is empty or none")
        
        documents = document_loader(file_path)
        
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size = parent_chunk_size,
                                                chunk_overlap=parent_chunk_overlap,
                                                length_function=len)
        
        child_splitter = RecursiveCharacterTextSplitter(chunk_size = child_chunk_size,
                                                chunk_overlap=child_chunk_overlap,
                                                length_function=len)
        
        embedding_model = embeddings()
        
        vectore_store = Chroma(
            collection_name="split_parent", embedding_function=embedding_model
        )
        
        doc_store = InMemoryStore()
        
        retriever = ParentDocumentRetriever(
            vectorstore=vectore_store,
            docstore=doc_store,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter
        )
        
        retriever.add_documents(documents=documents)
        return retriever
    
    except ValueError as e:
        print(f"value error: {e}")
        return None
    
    except Exception as e:
        print(f"unknown error: {e}")
        return None
    