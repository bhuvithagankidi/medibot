# C:/Users/bhuvi/PycharmProjects/medi-bot/tools.py

import logging
from typing import List
from langchain_core.documents import Document
from langchain.agents import tool

logger = logging.getLogger("MedicalTools")

def get_all_tools(retriever):
    """
    This function is a factory that creates and returns a list of specialized tools.
    It takes an initialized retriever and 'injects' it into the tools
    so they have access to the vector data.
    """

    def rag_query(prompt: str, data_type: str):
        """
        A helper function to perform a Retrieval-Augmented Generation query
        with a specific filter on the retriever.
        """
        if not retriever:
            logger.error("Retriever is not initialized. Cannot perform RAG query.")
            return [Document(page_content="⚠️ Error: The knowledge base retriever has not been set up correctly.")]
        try:
            retriever.search_kwargs["filter"] = {"type": data_type}
            docs = retriever.invoke(prompt)
            # --- This is the requested print statement ---
            #print(f"--- RAG Query Result for type '{data_type}' ---")
            #print(docs)
            #print("------------------------------------------")

            return docs

            #return retriever.invoke(prompt)
        except Exception as e:
            logger.exception(f"RAG query failed for data_type: {data_type}")
            return [Document(page_content=f"⚠️ An error occurred while querying about {data_type}.")]

    # The return types are changed from `str` to `List[Document]` to ensure the raw
    # document objects are passed to the agent's observation step for evaluation.
    # The agent is smart enough to process the string representation of these documents
    # in its thought process.

    @tool
    def DiagnosisTool(symptoms: str):
        """Useful for when you need to predict a disease based on a list of symptoms. The input should be a string describing the symptoms."""
        if not symptoms or len(symptoms.split()) < 2:
            # Return a Document object so the type is consistent.
            return [Document(page_content="Please provide more detailed symptoms for a better diagnosis. At least two symptoms are recommended.")]
        return rag_query(f"What disease is most likely indicated by these symptoms: {symptoms}", "diagnosis")

    @tool
    def SeverityTool(symptoms: str):
        """Useful for when you need to assess the severity of a given symptom or set of symptoms. The input should be a string of symptoms."""
        return rag_query(f"Based on the provided data, what is the severity of the following symptoms: {symptoms}", "severity")

    @tool
    def DescriptionTool(disease: str):
        """Useful for when you need a detailed description of a specific disease. The input should be the name of a single disease."""
        return rag_query(f"Provide a detailed description of the disease: {disease}", "description")

    @tool
    def PrecautionTool(disease: str):
        """Useful for when you need to know the recommended precautions for a specific disease. The input should be the name of a single disease."""
        return rag_query(f"What are the recommended precautions for someone with: {disease}", "precaution")

    return [DiagnosisTool, SeverityTool, DescriptionTool, PrecautionTool]