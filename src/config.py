import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(override=True)
 
class Config:
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    EMBEDDED_MODEL = "text-embedding-3-large"
    RAG_MODEL = "llama-3.1-8b-instant"
 
    DATA_DIR = Path("D:/PROJECT_LLMOPS/LEGAL_BOT/data")
   
    DOCUMENT_PATHS = [
        f"{DATA_DIR}/BNS.pdf",
        f"{DATA_DIR}/BSA.pdf",
        f"{DATA_DIR}/BNYS.pdf"
    ]