from langchain_ollama.llms import OllamaLLM

#define llm model
def llm_model (model_id):
    model_name = "llama3.1"
    
    llm = OllamaLLM(
        model=model_name
    )
    
    return llm


    
    
    
    