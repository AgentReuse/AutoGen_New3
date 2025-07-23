from Response_reuse import SemanticCache
from diskcache import Cache

semantic_cache = SemanticCache(
    embedding_model_path="./m3e-small",
    cache_path="./semantic_cache"
)
semantic_cache.save_to_cache("123","456","789")
semantic_cache.save_to_cache("789","123","741")