# Loading the data into vector DB by chunking teh documents and filter by type

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from google.colab import drive
import os

def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def load_medical_docs():
    docs = []

    # 1. Diagnosis (dataset.csv)
    #drive.mount("/content/gdrive")
    #diag_df = pd.read_csv("/content/gdrive/MyDrive/Datasets/Medibot/dataset.csv")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    file_path = os.path.join(DATA_DIR, "dataset.csv")
    diag_df = pd.read_csv(file_path)
    diagnosis_texts = []
    for _, row in diag_df.iterrows():
        symptoms = [str(row[f"Symptom_{i}"]) for i in range(1, 18) if pd.notnull(row.get(f"Symptom_{i}"))]
        if symptoms:
            diagnosis_texts.append(f"Disease: {row['Disease']} | Symptoms: {', '.join(symptoms)}")

    diagnosis_doc = Document(
        page_content="\n".join(diagnosis_texts),
        metadata={"type": "diagnosis"}
    )
    docs += chunk_documents([diagnosis_doc])

    # 2. Severity
    #sev = pd.read_csv("/content/gdrive/MyDrive/Datasets/Medibot/Symptom-severity.csv")
    file_path = os.path.join(DATA_DIR, "Symptom-severity.csv")
    sev = pd.read_csv(file_path)
    sev_texts = [f"Symptom: {row['Symptom']} → Severity Score: {row['weight']}" for _, row in sev.iterrows()]
    docs += chunk_documents([Document(page_content="\n".join(sev_texts), metadata={"type": "severity"})])

    # 3. Description
    #desc = pd.read_csv("/content/gdrive/MyDrive/Datasets/Medibot/symptom_Description.csv")
    file_path = os.path.join(DATA_DIR, "symptom_Description.csv")
    desc = pd.read_csv(file_path)
    desc_texts = [f"Disease: {row['Disease']} → Description: {row['Description']}" for _, row in desc.iterrows()]
    docs += chunk_documents([Document(page_content="\n".join(desc_texts), metadata={"type": "description"})])

    # 4. Precautions
    #prec = pd.read_csv("/content/gdrive/MyDrive/Datasets/Medibot/symptom_precaution.csv")
    file_path = os.path.join(DATA_DIR, "symptom_precaution.csv")
    prec = pd.read_csv(file_path)
    prec_texts = []
    for _, row in prec.iterrows():
        precautions = [str(row[col]) for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'] if pd.notnull(row[col])]
        prec_texts.append(f"Disease: {row['Disease']} → Precautions: {', '.join(precautions)}")

    docs += chunk_documents([Document(page_content="\n".join(prec_texts), metadata={"type": "precaution"})])

    return docs
