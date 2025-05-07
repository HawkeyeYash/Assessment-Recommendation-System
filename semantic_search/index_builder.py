from pathlib import Path
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from semantic_search.config import CSV_PATH, INDEX_PATH, EMBEDDING_DIM, GROQ_API_KEY
import pandas as pd

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
Settings.llm = Groq(api_key=GROQ_API_KEY, model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2, system_prompt="You are a helpful Assessment Recommendation Assistant. Help the user find the best assessment for their needs.")

df = pd.read_csv(CSV_PATH)
docs = []
for _, row in df.iterrows():
    chunks = [
        f"Assessment Name: {row.get('Assessment Name', '')}",
        f"Description: {row.get('Description', '')}",
        f"Duration: {row.get('Duration (min)', '')} minutes",
        f"Job Levels: {row.get('Job Levels', '')}",
    ]
    for chunk in chunks:
        docs.append(Document(text=chunk.strip(), metadata={"id": row.get("ID", "")}))

if Path(INDEX_PATH).exists():
    vector_store = FaissVectorStore.from_persist_dir(INDEX_PATH)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=INDEX_PATH)
    index = load_index_from_storage(storage_context)
else:
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    index.storage_context.persist(persist_dir=INDEX_PATH)

vector_index = index
