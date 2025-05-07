import pandas as pd
import sqlite3
from semantic_search.config import CSV_PATH, DB_PATH

df = pd.read_csv(CSV_PATH)
conn = sqlite3.connect(DB_PATH)
df.to_sql("assessments", conn, if_exists="replace", index=False)
