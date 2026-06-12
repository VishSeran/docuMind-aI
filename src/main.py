from src.utilities import llm_model,retriever_call


def retriever_qa (file_path, query):
    
    try:
        llm = llm_model()
        retriever_obj = retriever_call(file_path)
        
        
    except Exception as e:
        print(f"unknown error:{e}")
        return None