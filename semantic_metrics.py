from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Union

class SemanticEvaluator:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        
        embeddings1 = self.model.encode(text1, convert_to_tensor=True)
        embeddings2 = self.model.encode(text2, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        return float(cosine_scores[0][0])

    def compare_batch(self, reference: str, candidates: List[str]) -> List[float]:
        ref_emb = self.model.encode(reference, convert_to_tensor=True)
        cand_embs = self.model.encode(candidates, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(ref_emb, cand_embs)
        return cosine_scores[0].tolist()
