"""Top-k semantic retrieval over the FAISS index."""
from ragqa import config
from ragqa import index as index_mod


class Retriever:
    def __init__(self, index, chunks, model_name, meta=None):
        self.index = index
        self.chunks = chunks
        self.model_name = model_name
        self.meta = meta or {}

    @classmethod
    def from_dir(cls, index_dir=None):
        index, chunks, meta = index_mod.load_index(index_dir or config.INDEX_DIR)
        return cls(index, chunks, meta["embed_model"], meta)

    def search(self, query, k=None):
        k = k or config.TOP_K
        if not query.strip():
            return []
        vector = index_mod.embed([query], self.model_name)
        scores, ids = self.index.search(vector, min(k, self.index.ntotal))
        hits = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            chunk = self.chunks[int(i)]
            hits.append({
                "chunk_id": chunk["id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": round(float(score), 4),
            })
        return hits
