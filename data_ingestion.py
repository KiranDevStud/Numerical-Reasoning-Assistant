import os
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file.")

DATA_PATH = r"gsm8k/main/train-00000-of-00001.parquet"
PERSIST_DIRECTORY = r"chroma_db"
COLLECTION_NAME = "math_reasoning"

def load_data(file_path):
    """Loads GSM8K data from parquet file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    
    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    # GSM8K usually has 'question' and 'answer' columns
    if 'question' not in df.columns or 'answer' not in df.columns:
         # Fallback or check columns
        print(f"Warning: Expected columns 'question', 'answer'. Found: {df.columns}")
        # Identify text columns if different
    return df

def create_documents(df):
    """Converts dataframe rows to LangChain Documents."""
    documents = []
    # Limit to a subset for testing/speed if needed, or process all. 
    # Processing all might take time depending on API limits. 
    # Let's start with first 500 for demonstration/speed. 
    # User can change this limit.
    limit = 500 
    print(f"Processing first {limit} rows...")
    
    for _, row in df.head(limit).iterrows():
        question = row['question']
        answer = row['answer']
        
        doc = Document(
            page_content=f"Question: {question}\nAnswer: {answer}",
            metadata={"source": "gsm8k", "type": "example"}
        )
        documents.append(doc)
    return documents

import time

def ingest_data():
    """Main ingestion function."""
    # 1. Load Data
    df = load_data(DATA_PATH)
    
    # 2. Create Documents
    docs = create_documents(df)
    print(f"Created {len(docs)} documents.")

    # 3. Initialize Embeddings
    print("Initializing Gemini Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 4. Create/Update Vector Store in Batches
    print(f"Persisting to {PERSIST_DIRECTORY}...")
    
    batch_size = 10  # Smaller batch size to avoid rate limits
    total_docs = len(docs)
    
    for i in range(0, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        print(f"Processing batch {i}/{total_docs}...")
        
        # Retry loop for rate limits
        for attempt in range(3):
            try:
                Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIRECTORY,
                    collection_name=COLLECTION_NAME
                )
                time.sleep(5)  # Longer sleep to respect rate limits
                break  # Success, move to next batch
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    wait_time = 30 * (attempt + 1)
                    print(f"Rate limit hit at batch {i}. Waiting {wait_time}s... (Attempt {attempt+1}/3)")
                    time.sleep(wait_time)
                else:
                    print(f"Error processing batch {i}: {e}")
                    time.sleep(10)
                    break # Critical error, move on or stop
            
    print("Data ingestion complete!")

if __name__ == "__main__":
    ingest_data()
