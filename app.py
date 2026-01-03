# app.py
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from schemas.classification_response import ClassificationResponse
import uuid
from services.fusion_service import MultimodalDocumentClassifier
import shutil
import os

NLP_KEYWORDS_PATH       = os.getenv("NLP_KEYWORDS_PATH" , "data/keywords.txt") 
VISION_MODEL_IDENTIFIER = os.getenv("VISION_MODEL_IDENTIFIER" , "google/vit-base-patch16-224-in21k")
VISION_REFERENCES       = os.getenv("VISION_REFERENCES" , "data/references")
UPLOADS_PATH            = os.getenv("UPLOADS_PATH" , "data/uploads")
app = FastAPI()

def get_classifier() :
    """
    Dependency that provides an instance of MultimodalDocumentClassifier
    configured with environment variables or default values.
    """
    classifier = MultimodalDocumentClassifier(
        nlp_keywords_path     = NLP_KEYWORDS_PATH , 
        nlp_keyword_threshold = 5 , 
        vision_references_dir = VISION_REFERENCES , 
        vision_model_name     = VISION_MODEL_IDENTIFIER , 
        fusion_weight_nlp     = 0.5
    )
    return classifier

@app.post("/classify-document", response_model=ClassificationResponse)
def classify_document(
    file: UploadFile = File(...),
    classifier: MultimodalDocumentClassifier = Depends(get_classifier),
):
    # ---- basic validation ----
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # ---- safe, unique filename ----
    ext = os.path.splitext(file.filename)[1]

    unique_name = f"{uuid.uuid4().hex[:4]}{ext}"

    filepath = os.path.join(UPLOADS_PATH, unique_name)

    # ---- write file synchronously (streamed) ----
    try:

        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    except OSError as e:

        raise HTTPException(status_code=500, detail="Failed to save file") from e
    
    finally:
        file.file.close()

    # ---- synchronous inference ----
    try:
        result = classifier.process_document(pdf_path=filepath)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Document processing failed") from e

    return result


# ...existing code...

@app.get("/healthcheck")
async def healthcheck():
    """
    Healthcheck endpoint to verify the service is running.
    """
    return {"status": "ok"}

