from src.utilities import llm_model, retriever_call
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


def retriever_qa(file_path, query):

    try:
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
        
        response = qa_chain.invoke(input={"input":query})
        return response['answer']
    
    except ValueError as e:
        print(f"value error: {e}")
        return None

    except Exception as e:
        print(f"unknown error:{e}")
        return None
