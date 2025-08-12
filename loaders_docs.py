import pandas as pd
from langchain.schema import Document
import os
import logging

# Setup logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_medical_docs() -> list[Document]:
    """
    Loads medical data from various CSV files, transforming each row into a
    separate, structured Document object with appropriate metadata. This approach
    creates semantically meaningful chunks for effective retrieval.

    Returns:
        A list of Document objects, one for each piece of medical knowledge.
    """
    docs = []
    logging.info("Starting to load and process medical documents into structured format...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # 1. Diagnosis (dataset.csv)
    try:
        diag_path = os.path.join(DATA_DIR, "dataset.csv")
        diag_df = pd.read_csv(diag_path)
        for _, row in diag_df.iterrows():
            # Collect all non-null symptoms for the current disease
            symptoms = [str(row[f"Symptom_{i}"]) for i in range(1, 18) if pd.notna(row.get(f"Symptom_{i}"))]
            if symptoms:
                # Create a clean, readable text block for the document content
                page_content = f"The disease '{row['Disease']}' is associated with the following symptoms: {', '.join(symptoms)}."
                # Create a document with specific metadata for filtered searches
                doc = Document(
                    page_content=page_content,
                    metadata={"type": "diagnosis", "disease": row['Disease']}
                )
                docs.append(doc)
        logging.info(f"Processed {len(diag_df)} diagnosis entries.")
    except FileNotFoundError:
        logging.error(f"Diagnosis file not found at {diag_path}. Skipping.")

    # 2. Severity (Symptom-severity.csv)
    try:
        sev_path = os.path.join(DATA_DIR, "Symptom-severity.csv")
        sev_df = pd.read_csv(sev_path)
        for _, row in sev_df.iterrows():
            page_content = f"The symptom '{row['Symptom']}' has a severity weight of {row['weight']}."
            doc = Document(
                page_content=page_content,
                metadata={"type": "severity", "symptom": row['Symptom']}
            )
            docs.append(doc)
        logging.info(f"Processed {len(sev_df)} severity entries.")
    except FileNotFoundError:
        logging.warning(f"Severity file not found at {sev_path}. Skipping.")

    # 3. Description (symptom_Description.csv)
    try:
        desc_path = os.path.join(DATA_DIR, "symptom_Description.csv")
        desc_df = pd.read_csv(desc_path)
        for _, row in desc_df.iterrows():
            page_content = f"Description of {row['Disease']}: {row['Description']}"
            doc = Document(
                page_content=page_content,
                metadata={"type": "description", "disease": row['Disease']}
            )
            docs.append(doc)
        logging.info(f"Processed {len(desc_df)} disease descriptions.")
    except FileNotFoundError:
        logging.warning(f"Description file not found at {desc_path}. Skipping.")

    # 4. Precautions (symptom_precaution.csv)
    try:
        prec_path = os.path.join(DATA_DIR, "symptom_precaution.csv")
        prec_df = pd.read_csv(prec_path)
        for _, row in prec_df.iterrows():
            precautions = [str(row[col]) for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'] if pd.notna(row.get(col))]
            if precautions:
                page_content = f"Recommended precautions for '{row['Disease']}': {', '.join(precautions)}."
                doc = Document(
                    page_content=page_content,
                    metadata={"type": "precaution", "disease": row['Disease']}
                )
                docs.append(doc)
        logging.info(f"Processed {len(prec_df)} precaution entries.")
    except FileNotFoundError:
        logging.warning(f"Precaution file not found at {prec_path}. Skipping.")

    logging.info(f"Total structured documents created: {len(docs)}")
    return docs