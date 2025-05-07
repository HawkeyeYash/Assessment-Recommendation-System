from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from semantic_search.query_engine import semantic_query

import os
import sqlite3
from fastapi import status
from semantic_search.config import DB_PATH, INDEX_PATH, GROQ_API_KEY
from semantic_search.index_builder import vector_index
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
async def status_check():
    try:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set")

        if not os.path.exists(DB_PATH) or not os.path.isdir(INDEX_PATH):
            raise FileNotFoundError("Required files/directories missing")

        _ = vector_index.as_query_engine().query("test")

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("SELECT 1")

        return {"status": "healthy"}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service Unavailable")

class SearchRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: SearchRequest):
    try:
        return semantic_query(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
