import numpy as np
from skimage.feature import hog as skimage_hog
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize


class HOGFeatureExtractor : 

    def __init__(self, resize = (128 , 128),
                gradient_orientations=9,
                pixels_per_cell=(16 , 16) , 
                cells_per_bloc= (2 , 2)
                 ):


        self.resize = resize
        self.gradient_orientations= gradient_orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_bloc = cells_per_bloc

    def extract_features(self , image : Image.Image | np.ndarray ) : 

         # convert PIL image (if needed to numpy array) :
        if isinstance(image , Image.Image) : 
            image = np.array(image) 

        # convert image to grayscale :
        if len(image.shape) == 3 :

            if image.shape[2] == 4 : # rgba image
                image = color.rgb2gray(image[: , : , : 3])
            
            elif image.shape[2] == 3 :
                image = color.rgb2gray(image) 
            
            else :
                raise
        
        # resize the image to a standard size :
        image = resize(image , self.resize , anti_aliasing = True) 

        # extract hog features :
        features, hog_image = skimage_hog(
            image , 
            orientations= self.gradient_orientations, # number of orientation bins
            pixels_per_cell= self.pixels_per_cell ,  # size of a cell 
            cells_per_block= self.cells_per_bloc ,
            block_norm='L2-Hys' , # bloc normalization startegy
            visualize=True, 
            feature_vector=True, 
            channel_axis=None
        )


        return features
    



if __name__ == "__main__":
    # Initialize extractor
    extractor = HOGFeatureExtractor()
    
    # Load a sample image
    image = Image.open("data/images/test_image.png")
    
    # Extract features
    features = extractor.extract_features(image)
    
    print(f"Feature shape: {features.shape}")
    print(f"Feature vector (first 10): {features[:10]}")