from services.nlp_service import NLPModule
from services.vision_service import VisionModule
import numpy as np

class MultimodalDocumentClassifier:
       

    def __init__(self, 
                 nlp_keywords_path='data/keywords.txt', 
                 nlp_keyword_threshold=10, 
                 vision_references_dir='data/references',
                 fusion_weight_nlp=0.5 , 
                 strategy = "vit" ,  # hog , cnn , vit
                 classification_treshold = 0.5 , 
                 similarity_measure = "cosine"): # euclidean, manhattan
        


        self.nlp    = NLPModule(keywords_path=nlp_keywords_path, keyword_threshold=nlp_keyword_threshold)

        self.vision = VisionModule(references_dir=vision_references_dir , strategy=strategy)

        self.fusion_weight_nlp = fusion_weight_nlp

        self.classification_treshold = classification_treshold

        self.similarity_measure = similarity_measure


    def predict(self, pdf_path):
      
        nlp_result = self.nlp.process_document(pdf_path)

        vision_result = self.vision.predict(pdf_path , similarity_measure=self.similarity_measure)

        # Fusion: weighted average of normalized scores
        nlp_score = float(nlp_result.get('score', 0.0))

        vision_score = float(vision_result.get('score', 0.0))
        
        fused_score = float(self.fusion_weight_nlp * nlp_score + (1 - self.fusion_weight_nlp) * vision_score)

        # Simple rule: classify as 'transaction' if fused_score > 0.5
        prediction = 'transaction' if fused_score > self.classification_treshold else 'non-transaction'

        return {
            'prediction': prediction,
            'fused_score': fused_score,
            'nlp_result': nlp_result,
            'vision_result': vision_result
        }




if __name__ == "__main__" : 

    model = MultimodalDocumentClassifier(strategy="vit" , similarity_measure="cosine")

    result = model.predict("data/test/document_1.pdf")

    print(result)