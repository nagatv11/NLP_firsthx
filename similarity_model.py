from sentence_transformers import SentenceTransformer, util

class SymptomSimilarityModel:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

    def predict(self, input_text, known_symptoms, top_k=5):
        input_embedding = self.model.encode(input_text, convert_to_tensor=True)
        symptom_embeddings = self.model.encode(known_symptoms, convert_to_tensor=True)
        similarities = util.cos_sim(input_embedding, symptom_embeddings)[0]
        top_results = torch.topk(similarities, k=top_k)
        return [(known_symptoms[idx], similarities[idx].item()) for idx in top_results.indices]
