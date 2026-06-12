from src.utilities import llm_model, retriever_call
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr


def retriever_qa(file_path, query):

    try:
        
        if isinstance(file_path,dict):
            file_path = file_path.get("path") or file_path.get("name")
        
        if not file_path:
            raise ValueError("file path in invalid or none")
        
        llm = llm_model()
        
        retriever_obj = retriever_call(file_path)

        prompt = ChatPromptTemplate.from_template("""
            
            Answer using only the context below.

            Context:
            {context}

            Question:
            {input}
            """)
        
        doc_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)
        
        #connect retriever
        qa_chain = create_retrieval_chain(retriever=retriever_obj,
                                          combine_docs_chain=doc_chain)
        
        response = qa_chain.invoke({"input":query})
        return response.get("answer") or response.get("result") or str(response)
    
    except ValueError as e:
        print(f"value error: {e}")
        return None

    except Exception as e:
        print(f"unknown error:{e}")
        return None
    
def gradio_interface():

    interface = gr.Interface(
        fn=retriever_qa,
        inputs=[
            gr.File(label="Upload PDF File", 
                    file_count="single", file_types=['.pdf'],type="filepath"),
            gr.TextArea(label="Input Query",placeholder="What's in your mind...")
        ],
        outputs= gr.TextArea(label="Response", placeholder="Waiting for response..."),
        title="DocuMind AI",
        description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
    )
    
    interface.launch(share=True)
    
if __name__ == "__main__":
    gradio_interface()