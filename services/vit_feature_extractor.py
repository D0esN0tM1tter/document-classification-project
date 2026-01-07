import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np


class ViTFeatureExtractor :

    def __init__(self , model_name : str = "google/vit-base-patch16-224"):

        self.device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
        self.model , self.processor = self._load_model(model_name=model_name)

    def extract_features(self , image : Image.Image | np.ndarray) -> np.ndarray: 

        if len(image.shape) == 2 :
                image = np.stack([image] * 3 , axis=-1)

        # convert numpy array to PIL object if needed :
        if isinstance(image , np.ndarray) :
            # Ensure correct format (0-255, uint8)
            if image.dtype in [np.float32, np.float64]:
                image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
                
            image = Image.fromarray(image)             
                    
        # process the image :
        inputs = self.processor(images=image , return_tensors = 'pt')
        inputs = {k : v.to(self.device) for k , v in inputs.items()}

        # extract features :
        with torch.no_grad() :
            outputs = self.model(**inputs) 
            features = outputs.last_hidden_state[: , 0 , :].squeeze().cpu().numpy()
        
        return features

    
    def _load_model(self , model_name : str) : 

        # load the model and processor : 
        processor = ViTImageProcessor.from_pretrained(model_name) 
        model     = ViTModel.from_pretrained(model_name).to(self.device)
        
        # set the model to evaluation mode :
        model.eval()

        return model , processor

    
    