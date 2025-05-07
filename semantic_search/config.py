import os

CSV_PATH = "shl_product_catalog_cleaned.csv"
DB_PATH = "assessments.db"
INDEX_PATH = "./vector_index_faiss"
EMBEDDING_DIM = 768

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
