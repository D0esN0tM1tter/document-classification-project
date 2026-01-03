from pydantic import BaseModel

class ClassificationResponse(BaseModel) :

    prediction    : str
    fused_score   : float
    nlp_result    : dict
    vision_result : dict