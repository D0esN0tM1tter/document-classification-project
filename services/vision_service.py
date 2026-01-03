
import os
import torch
import numpy as np
from transformers import ViTModel, ViTImageProcessor
from services.vision_preprocessing import VisionPreprocessor
from typing import List, Dict, Any, Union
from PIL import Image
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VisionService")

class VisionService:
    """
    VisionService encapsulates the workflow for document image classification using a Vision Transformer.
    It provides methods for model loading, reference vector creation, feature extraction, similarity computation,
    and a full document processing pipeline returning a rich result dictionary.
    """

    def __init__(self, references_dir : str , model_name: str = "google/vit-base-patch16-224-in21k"):

        self.model_name = model_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.processor = self.load_model()

        self.model.eval()

        self.preprocessor = VisionPreprocessor()

        self.references_dir = references_dir

        self.reference_vector = self.preprocess_references(references_dir=self.references_dir)
        

    def load_model(self):
        """
        Loads the ViT model and processor from Hugging Face.
        """
        model = ViTModel.from_pretrained(self.model_name).to(self.device)
        processor = ViTImageProcessor.from_pretrained(self.model_name) 

        return model , processor

    def preprocess_references(self, references_dir: str):
        """
        Produces encoded features for the 'banking_transaction' class based on reference documents in a directory.
        Args:
            references_dir: Directory containing PDF files representing banking transactions.
        Sets self.reference_vector = representative feature vector for banking transactions.
        """
        
        features = []

        pdf_files = [os.path.join(references_dir, f) for f in os.listdir(references_dir) if f.lower().endswith('.pdf')]

        if not pdf_files:
            logger.warning(f"No PDF files found in references_dir: {references_dir}")

        for pdf_path in pdf_files:

            try:
                images = self.preprocessor.convert_pdf_to_image(pdf_path)

                if not images:
                    logger.warning(f"No images extracted from PDF: {pdf_path}")

                for idx, img in enumerate(images):

                    try:
                        img_np = self.preprocessor.preprocess(img)

                        feat = self.extract_features(img_np)

                        if feat is None or (hasattr(feat, 'size') and feat.size == 0):

                            logger.warning(f"Feature extraction returned None/empty for PDF: {pdf_path}, image index: {idx}")
                        else:
                            features.append(feat)

                    except Exception as fe:
                        logger.error(f"Feature extraction failed for PDF: {pdf_path}, image index: {idx}, error: {fe}")

            except Exception as e:
                logger.error(f"Failed to process PDF: {pdf_path}, error: {e}")

        if not features:
            logger.error(f"No features extracted from any reference PDF in directory: {references_dir}")
            
        return np.mean(features, axis=0) if features else None

    def extract_features(self, image: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Performs forward pass through the transformer and extracts the encoder's result.
        Args:
            image: Preprocessed image as numpy array (H, W, C) or PIL Image
        Returns:
            np.ndarray: Feature vector from the transformer (CLS token)
        """


        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        return features

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Computes cosine similarity between two vectors.
        """
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def classify_image(self, image: Union[np.ndarray, Any]) -> Dict[str, float]:
        """
        Classifies the input image as 'banking_transaction' or 'not_banking_transaction' based on similarity to the reference vector.
        Args:
            image: Preprocessed image as numpy array or PIL Image
        Returns:
            Dict[str, float]: Dictionary with similarity score for 'banking_transaction'.
        """
        if self.reference_vector is None:
            raise ValueError("Reference vector for 'banking_transaction' not set. Run preprocess_references first.")
        feat = self.extract_features(image)
        score = self.compute_similarity(feat, self.reference_vector)
        return {"banking_transaction": score}

    def process_document(self, pdf_path: str) -> dict:
        """
        Encapsulates the full vision workflow: PDF to images, preprocessing, feature extraction, similarity scoring.
        Returns a rich dictionary with all relevant details and error handling.
       
        """

        result = {
            'score': None,
            'scores': [],
            'num_images': 0,
            'pdf_path': pdf_path,
            'reference_vector_set': self.reference_vector is not None,
            'error': None
        }

        if not result['reference_vector_set']:
            result['error'] = "Reference vector for 'banking_transaction' not set. Run preprocess_references first."
            return result
        
        try:

            images = self.preprocessor.convert_pdf_to_image(pdf_path)

            result['num_images'] = len(images)

            for img in images:

                try:
                    img_np = self.preprocessor.preprocess(img)

                    feat = self.extract_features(img_np)

                    score = self.compute_similarity(feat, self.reference_vector)

                    result['scores'].append(score)


                except Exception as page_e:

                    result['scores'].append(0.0)


            if result['scores']:

                result['score'] = float(np.mean(result['scores']))

            else:
                result['score'] = 0.0
        except Exception as e:
            result['error'] = str(e)
        return result