import sqlite3
from semantic_search.index_builder import vector_index
from semantic_search.config import DB_PATH

def semantic_query(prompt: str, max_results: int = 10, min_score: float = 0.8):
    query_engine = vector_index.as_query_engine(similarity_top_k=20)
    response = query_engine.query(prompt)

    scored_results = [
        (node.metadata["id"], node.score)
        for node in response.source_nodes
        if node.score is not None and node.score >= min_score
    ]

    if not scored_results:
        return []

    sorted_ids = [item[0] for item in sorted(scored_results, key=lambda x: -x[1])]

    with sqlite3.connect(DB_PATH) as conn:
        placeholders = ",".join("?" for _ in sorted_ids)
        sql = f"SELECT * FROM assessments WHERE ID IN ({placeholders})"
        cursor = conn.execute(sql, sorted_ids)
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]

    result_dict = {row[column_names.index("ID")]: dict(zip(column_names, row)) for row in rows}
    final_results = []
    for id_ in sorted_ids:
        if id_ in result_dict:
            result = result_dict[id_].copy()
            result.pop("ID", None)

            test_type = result.get("Test Type")
            if isinstance(test_type, str):
                result["Test Type"] = [t.strip() for t in test_type.split(",") if t.strip()]

            final_results.append(result)

    return final_results[:max_results]
