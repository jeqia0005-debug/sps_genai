import spacy


class EmbeddingModel:
    def __init__(self, model_name="en_core_web_lg"):
        self.model_name = model_name
        self.nlp = spacy.load(model_name)
    
    def calculate_embedding(self, word):
        return self.nlp(word).vector

    def calculate_similarity(self, word1, word2):
        return float(self.nlp(word1).similarity(self.nlp(word2)))