import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class SemanticCache:
    def __init__(self, embedding_model_path: str, cache_path: str):
        self.model = SentenceTransformer(embedding_model_path)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.vector_id_map = {}
        self.id_counter = 0

        # 使用sqlite作为缓存
        self.cache_db_path = os.path.join(cache_path, "semantic_cache.db")
        os.makedirs(cache_path, exist_ok=True)
        self.conn = sqlite3.connect(self.cache_db_path, check_same_thread=False)
        self._init_db()
        self._load_cache()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                query TEXT PRIMARY KEY,
                response TEXT,
                plan TEXT
            )
        ''')
        self.conn.commit()

    def _load_cache(self):
        print("加载历史语义缓存中...")
        cursor = self.conn.cursor()
        cursor.execute("SELECT query FROM cache")
        all_queries = cursor.fetchall()
        for (query,) in all_queries:
            vector = self.model.encode(query, normalize_embeddings=True).astype(np.float32)
            self.index.add(np.array([vector]))
            self.vector_id_map[self.id_counter] = query
            self.id_counter += 1
        print(f"已恢复 {self.id_counter} 条语义问答缓存\n")

    def get_embedding(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def search_similar_query(self, query_vector: np.ndarray):
        threshold = 0
        top_k = 1
        if self.index.ntotal == 0:
            return None, 0
        scores, indices = self.index.search(np.array([query_vector]), top_k)
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                matched_query = self.vector_id_map[idx]
                # 从缓存里直接读出
                cursor = self.conn.cursor()
                cursor.execute("SELECT response, plan FROM cache WHERE query=?", (matched_query,))
                row = cursor.fetchone()
                cached_data = {"response": row[0], "plan": row[1]} if row else {}
                return matched_query, score, cached_data
        return None, 0

    def save_to_cache(self, query: str, response: str = None, plan: str = None):
        if response is None and plan is None:
            print(f"[警告] 未传入 response 或 plan，跳过缓存保存：{query}")
            return
        cursor = self.conn.cursor()
        if response is not None and plan is not None:
            cursor.execute('''
                INSERT OR REPLACE INTO cache (query, response, plan)
                VALUES (?, ?, ?)
            ''', (query, response, plan))
        elif response is None and plan is not None:
            cursor.execute('''
                    INSERT INTO cache (query, plan) VALUES (?, ?)
                    ON CONFLICT(query) DO UPDATE SET plan=excluded.plan
                ''', (query, plan))
        elif response is not None and plan is None:
            cursor.execute('''
                INSERT INTO cache (query, response)
                VALUES (?, ?)
                ON CONFLICT(query) DO UPDATE SET response=excluded.response
            ''', (query, response))
        self.conn.commit()

        vector = self.get_embedding(query)
        self.index.add(np.array([vector]))
        self.vector_id_map[self.id_counter] = query
        self.id_counter += 1

        cursor.execute('SELECT * FROM cache WHERE query=?', (query,))
        row = cursor.fetchone()
        print(f"[DEBUG] 数据库中的条目: {row}")

    def extract_plan(self, response_text: str) -> str:
        return response_text.split("。")[0] + "。" if "。" in response_text else response_text

    def close(self):
        self.conn.close()

# 用法示例
if __name__ == "__main__":
    cache = SemanticCache("./m3e-small", "./cache_sqlite")
    cache.save_to_cache("你好吗", response="我很好。", plan="打招呼")
    # cache.save_to_cache("天气如何", response="今天天气不错。")

    # 检索
    query = "你最近怎么样"
    vec = cache.get_embedding(query)
    # result = cache.search_similar_query(vec)
    # print(result)
