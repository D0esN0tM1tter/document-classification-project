from services.nlp_service import NLPModule
from services.vision_service import VisionService

class MultimodalDocumentClassifier:
       
    """
    Combines NLP and Vision predictions for document classification.
    """

    def __init__(self, 
                 nlp_keywords_path='data/keywords.txt', 
                 nlp_keyword_threshold=5, 
                 vision_references_dir='data/references',
                 vision_model_name="google/vit-base-patch16-224-in21k",
                 fusion_weight_nlp=0.5):
        
        """
        Args:
            nlp_keywords_path: Path to NLP keywords file.
            nlp_keyword_threshold: Threshold for NLP classification.
            vision_references_dir: Directory for vision reference PDFs.
            vision_model_name: HuggingFace model name for vision.
            fusion_weight_nlp: Weight for NLP score in fusion (0-1).
        """

        self.nlp    = NLPModule(keywords_path=nlp_keywords_path, keyword_threshold=nlp_keyword_threshold)

        self.vision = VisionService(references_dir=vision_references_dir, model_name=vision_model_name)

        self.fusion_weight_nlp = fusion_weight_nlp


    def process_document(self, pdf_path):
        """
        Runs both NLP and Vision pipelines, combines results.
        Args:
            pdf_path: Path to PDF file.
        Returns:
            dict: Combined result with individual and fused scores/predictions.
        """

        nlp_result = self.nlp.process_document(pdf_path)

        vision_result = self.vision.process_document(pdf_path)

        # Fusion: weighted average of normalized scores
        nlp_score = nlp_result.get('score', 0.0)

        vision_score = vision_result.get('score', 0.0)
        
        fused_score = self.fusion_weight_nlp * nlp_score + (1 - self.fusion_weight_nlp) * vision_score

        # Simple rule: classify as 'transaction' if fused_score > 0.5
        prediction = 'transaction' if fused_score > 0.5 else 'non-transaction'

        return {
            'prediction': prediction,
            'fused_score': fused_score,
            'nlp_result': nlp_result,
            'vision_result': vision_result
        }

