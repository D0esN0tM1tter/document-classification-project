import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class CNNFeatureExtractor:
    
    def __init__(self, model_name: str = "resnet50"):
        """
        Initialize CNN feature extractor
        
        Args:
            model_name: Name of the pretrained model 
                       ('resnet50', 'resnet101', 'vgg16', 'efficientnet_b0')
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_name=model_name)
        self.transform = self._get_transforms()
    
    def extract_features(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Extract features from an image"""
        # Convert numpy array to PIL object if needed
        if isinstance(image, np.ndarray):
            # Ensure correct format (0-255, uint8)
            if image.dtype in [np.float32, np.float64]:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            
            # Handle grayscale images
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            
            image = Image.fromarray(image)
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)
            features = features.squeeze().cpu().numpy()
        
        return features
    
    def _load_model(self, model_name: str):
        """Load pretrained CNN model and remove classification head"""
        
        # load CNN model : 
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final classification layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        
        # Move to device and set to evaluation mode
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _get_transforms(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


if __name__ == "__main__":
    # Initialize extractor
    extractor = CNNFeatureExtractor(model_name="resnet50")
    
    # Load a sample image
    image = Image.open("data/images/test_image.png")
    
    # Extract features
    features = extractor.extract_features(image)
    
    print(f"Feature shape: {features.shape}")
    print(f"Feature vector (first 10): {features[:10]}")