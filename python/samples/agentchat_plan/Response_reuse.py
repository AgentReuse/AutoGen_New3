import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from diskcache import Index

class SemanticCache:
    def __init__(self, embedding_model_path: str, cache_path: str):
        self.model = SentenceTransformer("./m3e-small")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.vector_id_map = {}
        self.id_counter = 0
        self.cache = Index('./semantic_cache')
        self._load_cache()

    def _load_cache(self):
        print("加载历史语义缓存中...")
        for key in self.cache:
            vector = self.model.encode(key, normalize_embeddings=True).astype(np.float32)
            self.index.add(np.array([vector]))
            self.vector_id_map[self.id_counter] = key
            self.id_counter += 1
        print(f"已恢复 {self.id_counter} 条语义问答缓存\n")

    def get_embedding(self, text: str) -> np.ndarray:           #向量化，传入文本即可返回faiss适用的向量格式
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def search_similar_query(self, query_vector: np.ndarray):    #faiss的indexflatip进行向量化搜索并返回最相似的问题和相似度
        threshold = 0    #初始阈值
        top_k = 1        #找相似度最高的top_k条

        if self.index.ntotal == 0:
            return None

        scores, indices = self.index.search(np.array([query_vector]), top_k)
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                matched_query = self.vector_id_map[idx]
                return matched_query, score
        return None

    def save_to_cache(self, query: str, response: str = None, plan: str = None):
        entry = {}
        if response is not None:
            entry["response"] = response
        if plan is not None:
            entry["plan"] = plan

        if not entry:
            print(f"[警告] 未传入 response 或 plan，跳过缓存保存：{query}")
            return

        self.cache[query] = entry

        vector = self.get_embedding(query)
        self.index.add(np.array([vector]))
        self.vector_id_map[self.id_counter] = query
        self.id_counter += 1

    def xtract_plane(self, response_text: str) -> str:
        return response_text.split("。")[0] + "。" if "。" in response_text else response_text

    '''
    async def ask_with_cache(self, model_client, query: str):
        query_vector = self.get_embedding(query)
        result = self.search_similar_query(query_vector)

        if result:
            matched_query, score, cached_data = result
            if score >= 0.90:
                print(f"响应复用，相似问题：{matched_query} (相似度: {score:.4f})")
                return cached_data["response"], 2
            elif 0.75 <= score < 0.90:
                print(f"计划复用，相似问题：{matched_query} (相似度: {score:.4f})")
                reused_plan = cached_data.get("plan", "[无计划]")
                return f"[计划复用] 来自问题：{matched_query}\n计划：{reused_plan}", 1

        try:
            from autogen_core.models import UserMessage
            response = await model_client.create([
                UserMessage(content=query, source="user")
            ])
            raw_output = response.content if hasattr(response, 'content') else str(response)
            plan = self.extract_plan(raw_output)
            self.save_to_cache(query, raw_output, plan)
            print("未命中，调用模型并缓存结果")
            return raw_output, 0

        except Exception as e:
            return f"[Autogen API 错误]：{str(e)}", 0
    '''