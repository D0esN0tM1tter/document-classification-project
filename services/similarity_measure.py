import numpy as np
from numpy.linalg import norm

class Similarity :


    def compute_similarity(self , features_1 : np.ndarray , features_2 : np.ndarray , strategy = 'cosine') : 


        self._validate_inputs(features_1 , features_2)
        
        if strategy == 'cosine' :
            return self.cosine(features_1 , features_2)  

        elif strategy == 'euclidean' : 
            return self.euclidean(features_1 , features_2) 

        elif strategy == 'manhattan' :
            return self.manhattan(features_1 , features_2) 

        else : 
            raise ValueError("Unsupported similarity measure.")
        
    

    def _validate_inputs(self , features_1 : np.ndarray , features_2 : np.ndarray) : 
        pass

    def cosine(self , features_1 : np.ndarray , features_2 : np.ndarray) -> float : 
        
        # normalize the input vectors
        features_1 = features_1 / norm(features_1) 
        features_2 = features_2 / norm(features_2) 
        return np.dot(features_1 , features_2)

    def euclidean(self , features_1 : np.ndarray , features_2 : np.ndarray) -> float : 
                
        distance = np.sqrt( np.sum(( features_1 - features_2 ) ** 2 ))  
        return 1 / (1 + distance)

    def manhattan(self , features_1 : np.ndarray , features_2 : np.ndarray) -> float: 
        distance = np.sum(np.abs(features_1 - features_2) )
        return 1 / (1 + distance)
    


