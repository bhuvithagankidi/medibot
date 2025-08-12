# C:/Users/bhuvi/PycharmProjects/medi-bot/eda.py
"""
Performs Exploratory Data Analysis (EDA) on the medical datasets.

This script loads the raw data, generates several visualizations to understand
data distribution and relationships, and saves them to the 'eda' directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Define Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "eda")  # <-- Define the output directory

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

DIAGNOSIS_FILE = os.path.join(DATA_DIR, "dataset.csv")
SEVERITY_FILE = os.path.join(DATA_DIR, "Symptom-severity.csv")


def analyze_and_visualize_data():
    """
    Performs EDA on the medical datasets and saves visualizations.
    """
    logging.info("--- Starting Data Analysis & Visualization ---")

    # --- 1. Load and Inspect the Main Diagnosis Dataset ---
    try:
        diag_df = pd.read_csv(DIAGNOSIS_FILE)
    except FileNotFoundError:
        logging.error(f"Diagnosis file not found at {DIAGNOSIS_FILE}. Aborting.")
        return

    logging.info("Diagnosis Dataset Info:")
    diag_df.info()

    logging.info("\nMissing values in diagnosis dataset:")
    print(diag_df.isnull().sum())

    # --- 2. Statistical Analysis: Disease Frequency ---
    plt.figure(figsize=(12, 8))
    sns.countplot(y='Disease', data=diag_df, order=diag_df['Disease'].value_counts().index, palette='viridis')
    plt.title('Frequency of Diseases in the Dataset')
    plt.xlabel('Count')
    plt.ylabel('Disease')
    plt.tight_layout()
    # Save the plot to the 'eda' directory
    plt.savefig(os.path.join(OUTPUT_DIR, 'disease_frequency.png'))
    logging.info(f"Saved 'disease_frequency.png' to '{OUTPUT_DIR}'")
    plt.close()

    # --- 3. Statistical Analysis: Common Symptoms ---
    symptoms_df = diag_df.melt(id_vars=['Disease'], value_vars=[f'Symptom_{i}' for i in range(1, 18)])
    symptoms_df = symptoms_df.dropna().drop(columns=['variable'])
    symptoms_df.rename(columns={'value': 'Symptom'}, inplace=True)

    common_symptoms = symptoms_df['Symptom'].value_counts().head(20)
    logging.info("\nTop 20 most common symptoms across all diseases:")
    print(common_symptoms)

    plt.figure(figsize=(12, 8))
    sns.barplot(y=common_symptoms.index, x=common_symptoms.values, palette='plasma')
    plt.title('Top 20 Most Common Symptoms')
    plt.xlabel('Number of Diseases Sharing the Symptom')
    plt.ylabel('Symptom')
    plt.tight_layout()
    # Save the plot to the 'eda' directory
    plt.savefig(os.path.join(OUTPUT_DIR, 'common_symptoms.png'))
    logging.info(f"Saved 'common_symptoms.png' to '{OUTPUT_DIR}'")
    plt.close()

    # --- 4. Load and Analyze Severity Data ---
    try:
        severity_df = pd.read_csv(SEVERITY_FILE)
        logging.info("\nSeverity Dataset Info:")
        severity_df.info()

        plt.figure(figsize=(10, 6))
        sns.countplot(x='weight', data=severity_df, palette='coolwarm')
        plt.title('Distribution of Symptom Severity Scores')
        plt.xlabel('Severity Weight')
        plt.ylabel('Count of Symptoms')
        # Save the plot to the 'eda' directory
        plt.savefig(os.path.join(OUTPUT_DIR, 'severity_distribution.png'))
        logging.info(f"Saved 'severity_distribution.png' to '{OUTPUT_DIR}'")
        plt.close()

    except FileNotFoundError:
        logging.warning(f"Severity file not found at {SEVERITY_FILE}. Skipping severity analysis.")

    logging.info("--- Data Analysis Complete ---")


if __name__ == '__main__':
    analyze_and_visualize_data()