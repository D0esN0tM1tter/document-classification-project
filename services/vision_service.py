import numpy as np
import os
from services.vision_preprocessing import VisionPreprocessor
from PIL import Image
from services.vit_feature_extractor import ViTFeatureExtractor
from services.cnn_feature_extractor import CNNFeatureExtractor
from services.hog_feature_extractor import HOGFeatureExtractor
from services.similarity_measure import Similarity

class VisionModule :

    def __init__(self , references_dir : str = "data/references"  , strategy = "vit"):

        self.references_dir    = references_dir
        self.preprocessor      = VisionPreprocessor()
        self.similarity        = Similarity()

        # load the models to memeory only on demand :
        if strategy == "vit" : 
            self.feature_extractor = ViTFeatureExtractor()

        elif strategy == "cnn" :
            self.feature_extractor = CNNFeatureExtractor()

        elif strategy == "hog" : 
            self.feature_extractor = HOGFeatureExtractor() 
        
        else :
            raise ValueError("Unupported strategy.")

        self.references_vector = self.process_references(references_dir=self.references_dir)


    
    def process_document(self , document_path) ->  np.ndarray: 
        
        features_list = []
        images   = self.preprocessor.convert_pdf_to_image(pdf_path=document_path)

        for img in images :
            img_np   = self.preprocessor.preprocess(img)
            features = self.extract_features(image=img_np)
            features_list.append(features)

        if len(features_list) == 1 :
            return features_list[0]

        return np.mean(features_list , axis=0)


    def process_references(self , references_dir : str ) -> np.ndarray  :

        features_list = []
        pdf_files     = [os.path.join(references_dir , f ) for f in os.listdir(references_dir) if f.endswith('.pdf')]

        for pdf_file in pdf_files :
            features = self.process_document(document_path=pdf_file) 
            features_list.append(features)
        
        return np.mean(features_list , axis=0)


    def extract_features(self , image :Image.Image | np.ndarray) -> np.ndarray :
        # prepocess the input image (light preprocessing) :
        image    =  self.preprocessor.preprocess(image=image)
        return self.feature_extractor.extract_features(image)
    
    def predict(self , document_path : str , similarity_measure : str = "cosine") -> dict : 
        # preprocess and extract image features :
        features = self.process_document(document_path=document_path) 

        # compute similarity with reference documents : 
        score = self.similarity.compute_similarity(features_1=features , features_2=self.references_vector , strategy=similarity_measure) 

        return {'score' : float(score) , 'path' : document_path}
    


if __name__ == "__main__" : 

    module = VisionModule() 

    prediction = module.predict(document_path="data/test/document_2.pdf")

    print(prediction)

