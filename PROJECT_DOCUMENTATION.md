# Document Classification Project - Detailed Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [API Endpoints](#api-endpoints)
5. [Data Flow](#data-flow)
6. [Installation & Setup](#installation--setup)
7. [Configuration](#configuration)
8. [Testing](#testing)
9. [Deployment](#deployment)

---

## Project Overview

### Purpose
This project implements a **multimodal document classification system** designed to classify PDF documents, specifically distinguishing between banking transaction documents and non-transaction documents. The system leverages both **Natural Language Processing (NLP)** and **Computer Vision** techniques to achieve robust and accurate classification.

### Key Features
- **Multimodal Analysis**: Combines text analysis (NLP) and visual analysis (Vision Transformer) for comprehensive document understanding
- **REST API**: FastAPI-based web service for easy integration
- **Flexible Configuration**: Environment-based configuration for different deployment scenarios
- **French Language Support**: Optimized for French documents with specialized NLP processing
- **OCR Capability**: Fallback to OCR when PDFs don't contain extractable text
- **Reference-Based Classification**: Vision system uses reference documents to learn transaction patterns
- **Weighted Fusion**: Configurable weighting between NLP and vision predictions

### Use Case
The primary use case is automated classification of banking documents to identify transaction-related documents such as bank statements, receipts, and transaction confirmations, distinguishing them from other documents like letters, reports, or general correspondence.

---

## Architecture

### High-Level Architecture

```
┌──────────────┐
│   Client     │
│  (Upload PDF)│
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│         FastAPI Application          │
│            (app.py)                  │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│   MultimodalDocumentClassifier       │
│      (fusion_service.py)             │
└──────┬───────────────────────────────┘
       │
       ├────────────────┬───────────────┤
       ▼                ▼               ▼
┌─────────────┐  ┌──────────────┐  ┌─────────┐
│ NLP Module  │  │Vision Service│  │ Fusion  │
│(nlp_service)│  │(vision_*)    │  │ Logic   │
└─────────────┘  └──────────────┘  └─────────┘
       │                ▼
       ▼         ┌──────────────────┐
┌─────────────┐  │ VisionPreprocessor│
│Text Extract │  │ (preprocessing)  │
│  + OCR      │  └──────────────────┘
└─────────────┘
       │                │
       ▼                ▼
┌──────────────────────────────────────┐
│      Classification Result           │
│  {prediction, scores, details}       │
└──────────────────────────────────────┘
```

### Technology Stack

#### Backend Framework
- **FastAPI**: Modern, high-performance web framework
- **Uvicorn**: ASGI server for running the application
- **Pydantic**: Data validation using Python type annotations

#### NLP Components
- **PyPDF2**: Text extraction from PDF documents
- **pytesseract**: OCR for image-based PDFs
- **pdf2image**: PDF to image conversion
- **NLTK**: Natural language processing toolkit
  - French tokenization
  - Snowball French stemmer
  - Text preprocessing

#### Vision Components
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Pre-trained Vision Transformer models
- **ViT (Vision Transformer)**: Image classification model (`google/vit-base-patch16-224-in21k`)
- **OpenCV**: Image preprocessing
- **Pillow**: Image manipulation
- **NumPy**: Numerical operations

#### Development & Deployment
- **Docker**: Containerization
- **Python 3.10**: Programming language
- **Git**: Version control

---

## Core Components

### 1. FastAPI Application (`app.py`)

**Purpose**: Entry point for the application, defines REST API endpoints and handles HTTP requests.

**Key Responsibilities**:
- Endpoint definition and routing
- File upload handling with validation
- Unique filename generation to prevent conflicts
- Dependency injection for classifier instance
- Error handling and HTTP exception management
- Health check endpoint

**Main Endpoint**:
```python
POST /classify-document
- Accepts: PDF file upload
- Returns: Classification result with prediction, scores, and details
```

**Healthcheck Endpoint**:
```python
GET /healthcheck
- Returns: Service status
```

**Configuration Variables**:
- `NLP_KEYWORDS_PATH`: Path to keywords file (default: `data/keywords.txt`)
- `VISION_MODEL_IDENTIFIER`: Hugging Face model name (default: `google/vit-base-patch16-224-in21k`)
- `VISION_REFERENCES`: Reference documents directory (default: `data/references`)
- `UPLOADS_PATH`: Upload directory (default: `data/uploads`)

---

### 2. Fusion Service (`services/fusion_service.py`)

**Purpose**: Orchestrates the multimodal classification by combining NLP and Vision predictions.

**Class**: `MultimodalDocumentClassifier`

**Key Responsibilities**:
- Initialize and configure NLP and Vision modules
- Execute both classification pipelines
- Combine results using weighted fusion
- Produce final classification decision

**Fusion Algorithm**:
```python
fused_score = (weight_nlp × nlp_score) + (weight_vision × vision_score)
prediction = 'transaction' if fused_score > 0.5 else 'non-transaction'
```

**Configuration Parameters**:
- `nlp_keywords_path`: Path to keyword dictionary
- `nlp_keyword_threshold`: Minimum keywords for transaction classification
- `vision_references_dir`: Directory containing reference transaction PDFs
- `vision_model_name`: Hugging Face model identifier
- `fusion_weight_nlp`: Weight for NLP score (0-1), vision weight = 1 - nlp_weight

**Output Structure**:
```python
{
    'prediction': str,           # 'transaction' or 'non-transaction'
    'fused_score': float,        # Combined score [0, 1]
    'nlp_result': dict,          # NLP module output
    'vision_result': dict        # Vision module output
}
```

---

### 3. NLP Service (`services/nlp_service.py`)

**Purpose**: Text-based document classification using keyword matching and French language processing.

**Class**: `NLPModule`

**Key Features**:
- French language support with specialized stemmer
- Keyword-based classification
- Multiple text extraction methods (PyPDF2 + OCR fallback)
- Text preprocessing and normalization
- Score normalization

**Text Extraction Pipeline**:
1. **Primary**: PyPDF2 for text-based PDFs
2. **Fallback**: Tesseract OCR for scanned/image-based PDFs
3. **Threshold**: If <50 characters extracted, automatically fall back to OCR

**Text Preprocessing Steps**:
1. Convert to lowercase
2. Remove special characters (preserve French accents and numbers)
3. Tokenize using NLTK French tokenizer
4. Stem using French Snowball Stemmer

**Keyword Matching Algorithm**:
1. Load keywords from external file
2. Stem both document tokens and keywords
3. Count matching stemmed keywords
4. Compare count against threshold
5. Normalize score: `score = matches / total_keywords`

**Output Structure**:
```python
{
    'prediction': str,              # 'transaction' or 'non-transaction'
    'score': float,                 # Normalized score [0, 1]
    'matched_keywords': list,       # Keywords found in document
    'raw_count': int,               # Number of keyword matches
    'max_possible': int             # Total keywords in dictionary
}
```

**Dependencies**:
- Keywords file (`data/keywords.txt`): One keyword per line
- NLTK data: `punkt` tokenizer
- Tesseract OCR: System dependency (for OCR fallback)

---

### 4. Vision Service (`services/vision_service.py`)

**Purpose**: Visual document classification using deep learning and Vision Transformers.

**Class**: `VisionService`

**Key Features**:
- Vision Transformer (ViT) based feature extraction
- Reference-based similarity comparison
- GPU acceleration support
- Multi-page PDF processing
- Robust error handling

**Workflow**:

1. **Model Loading**:
   - Load pre-trained ViT model from Hugging Face
   - Load corresponding image processor
   - Move model to GPU if available
   - Set model to evaluation mode

2. **Reference Processing** (Initialization):
   - Load all reference PDFs from references directory
   - Convert each reference PDF to images
   - Preprocess each image
   - Extract features using ViT
   - Compute mean feature vector as "transaction template"

3. **Document Classification**:
   - Convert input PDF to images
   - Preprocess each page
   - Extract features using ViT (CLS token)
   - Compute cosine similarity with reference vector
   - Average scores across all pages

**Feature Extraction**:
- Uses ViT's CLS token (classification token)
- CLS token: First position in last hidden state
- Provides semantic representation of entire image
- Dimensionality: Depends on model (typically 768 for base models)

**Similarity Metric**:
```python
cosine_similarity = dot(vector1, vector2) / (norm(vector1) × norm(vector2))
```

**Output Structure**:
```python
{
    'score': float,                    # Average similarity score [0, 1]
    'scores': list[float],             # Per-page scores
    'num_images': int,                 # Number of pages processed
    'pdf_path': str,                   # Input PDF path
    'reference_vector_set': bool,      # Whether references loaded
    'error': str or None               # Error message if any
}
```

**Device Handling**:
- Automatically detects CUDA availability
- Falls back to CPU if GPU not available
- All tensors moved to appropriate device

---

### 5. Vision Preprocessing (`services/vision_preprocessing.py`)

**Purpose**: Image preprocessing pipeline optimized for Vision Transformer input.

**Class**: `VisionPreprocessor`

**Key Features**:
- PDF to image conversion
- Image quality enhancement
- Document deskewing
- Format normalization

**Processing Pipeline**:

1. **PDF Conversion**:
   - Convert PDF pages to PIL Images
   - Resolution: 300 DPI (configurable)
   - Format: RGB
   - **Note**: Currently processes only first page for efficiency

2. **Denoising**:
   - Method: Non-local means denoising
   - Reduces image noise while preserving edges
   - Parameters optimized for document images

3. **Deskewing**:
   - Detects document rotation using moments
   - Computes minimum area rectangle
   - Applies affine transformation to straighten
   - Handles rotation angles using optimal quadrant selection

4. **Format Normalization**:
   - Ensures RGB format (3 channels)
   - Converts grayscale to RGB if needed
   - Output: NumPy array ready for ViT processor

**Configuration**:
- `dpi`: Resolution for PDF conversion (default: 300)
- `image_format`: Output color format (default: "RGB")

**Deskewing Algorithm**:
```python
1. Convert to grayscale
2. Find non-zero pixel coordinates
3. Compute minimum area rectangle
4. Extract rotation angle
5. Normalize angle to [-45, 45] range
6. Apply rotation transformation
7. Return deskewed image
```

---

### 6. Schemas (`schemas/classification_response.py`)

**Purpose**: Define API response structure using Pydantic models.

**Model**: `ClassificationResponse`

**Fields**:
- `prediction`: str - Final classification result
- `fused_score`: float - Combined confidence score
- `nlp_result`: dict - Complete NLP analysis output
- `vision_result`: dict - Complete vision analysis output

**Benefits**:
- Automatic data validation
- API documentation generation
- Type safety
- Serialization/deserialization

---

### 7. Evaluation Module (`services/evaluation.py`)

**Purpose**: Batch processing and model evaluation (placeholder for future implementation).

**Planned Functions**:
- `process_all_documents()`: Batch document classification
- `evaluate()`: Compare predictions against ground truth labels
- Performance metrics calculation

**Current Status**: Stub implementation, ready for extension

---

## API Endpoints

### POST `/classify-document`

**Description**: Upload and classify a PDF document.

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: File upload (PDF)

**Example**:
```bash
curl -X POST "http://localhost:8000/classify-document" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Response** (200 OK):
```json
{
  "prediction": "transaction",
  "fused_score": 0.73,
  "nlp_result": {
    "prediction": "transaction",
    "score": 0.65,
    "matched_keywords": ["virement", "compte", "montant"],
    "raw_count": 13,
    "max_possible": 20
  },
  "vision_result": {
    "score": 0.81,
    "scores": [0.81],
    "num_images": 1,
    "pdf_path": "data/uploads/abc123.pdf",
    "reference_vector_set": true,
    "error": null
  }
}
```

**Error Responses**:
- 400: Invalid filename
- 500: File save failed or processing error

---

### GET `/healthcheck`

**Description**: Verify service availability.

**Request**:
- Method: GET

**Example**:
```bash
curl -X GET "http://localhost:8000/healthcheck"
```

**Response** (200 OK):
```json
{
  "status": "ok"
}
```

---

## Data Flow

### Complete Classification Pipeline

```
1. Client Upload
   └─> PDF file sent via HTTP POST

2. File Handling (app.py)
   ├─> Generate unique filename
   ├─> Save to uploads directory
   └─> Pass path to classifier

3. Multimodal Classification (fusion_service.py)
   ├─> NLP Pipeline
   │   ├─> Extract text (PyPDF2/OCR)
   │   ├─> Preprocess (lowercase, stem, tokenize)
   │   ├─> Match keywords
   │   ├─> Calculate score
   │   └─> Return {prediction, score, matches}
   │
   └─> Vision Pipeline
       ├─> Convert PDF to images
       ├─> Preprocess (denoise, deskew)
       ├─> Extract features (ViT)
       ├─> Compute similarity with references
       ├─> Average multi-page scores
       └─> Return {score, scores, metadata}

4. Fusion
   ├─> Weighted combination of scores
   ├─> Final decision (threshold @ 0.5)
   └─> Package complete result

5. Response
   └─> JSON with prediction + all details
```

### Data Storage Structure

```
data/
├── keywords.txt              # NLP keyword dictionary
├── references/               # Vision reference documents
│   ├── transaction1.pdf
│   ├── transaction2.pdf
│   └── ...
├── uploads/                  # Temporary uploads
│   └── [unique-id].pdf
└── test/                     # Test documents
```

---

## Installation & Setup

### Prerequisites

**System Requirements**:
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster vision processing
- (Optional) Docker for containerized deployment

**External Dependencies**:
- Tesseract OCR (for OCR functionality)
  - Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-fra`
  - macOS: `brew install tesseract tesseract-lang`
  - Windows: Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Installation Steps

1. **Clone Repository**:
```bash
git clone <repository-url>
cd document-classfication-project
```

2. **Create Virtual Environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download NLTK Data**:
```bash
python -c "import nltk; nltk.download('punkt')"
```

5. **Prepare Data Directories**:
```bash
mkdir -p data/uploads
mkdir -p data/references
mkdir -p data/test
```

6. **Add Keywords**:
Create `data/keywords.txt` with relevant keywords (one per line):
```
virement
compte
montant
banque
transaction
...
```

7. **Add Reference Documents**:
Place sample transaction PDFs in `data/references/` for vision training

8. **Run Application**:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

9. **Access API**:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

---

## Configuration

### Environment Variables

All configuration can be customized via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `NLP_KEYWORDS_PATH` | Path to keyword dictionary file | `data/keywords.txt` |
| `VISION_MODEL_IDENTIFIER` | Hugging Face model name | `google/vit-base-patch16-224-in21k` |
| `VISION_REFERENCES` | Directory with reference PDFs | `data/references` |
| `UPLOADS_PATH` | Directory for uploaded files | `data/uploads` |

**Example**:
```bash
export NLP_KEYWORDS_PATH="config/custom_keywords.txt"
export VISION_MODEL_IDENTIFIER="google/vit-large-patch16-224"
export FUSION_WEIGHT_NLP="0.6"
uvicorn app:app --port 8000
```

### Tuning Parameters

**NLP Module**:
- `keyword_threshold`: Minimum keyword matches for 'transaction' (default: 5)
- Adjust based on keyword dictionary size and document types

**Vision Module**:
- `model_name`: Choose different ViT variants for speed/accuracy trade-off
  - `vit-base-patch16-224`: Balanced (default)
  - `vit-large-patch16-224`: Higher accuracy, slower
  - `vit-small-patch16-224`: Faster, lower accuracy

**Fusion**:
- `fusion_weight_nlp`: Balance between NLP and vision (0-1)
  - 0.5: Equal weight (default)
  - >0.5: Trust NLP more
  - <0.5: Trust vision more

---

## Testing

### Test Structure

```
tests/
├── test_nlp_module.py       # NLP service tests
├── test_vision_module.py    # Vision service tests
└── test_fusion_module.py    # Integration tests
```

### Running Tests

**All Tests**:
```bash
pytest tests/
```

**Specific Module**:
```bash
pytest tests/test_nlp_module.py -v
```

**With Coverage**:
```bash
pytest tests/ --cov=services --cov-report=html
```

### Test Data

Place test PDFs in `data/test/` for testing purposes.

---

## Deployment

### Docker Deployment

**Build Image**:
```bash
docker build -t document-classifier .
```

**Run Container**:
```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e VISION_MODEL_IDENTIFIER="google/vit-base-patch16-224-in21k" \
  --name doc-classifier \
  document-classifier
```

**Docker Compose** (create `docker-compose.yml`):
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - NLP_KEYWORDS_PATH=data/keywords.txt
      - VISION_REFERENCES=data/references
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

### Production Considerations

1. **Performance**:
   - Use GPU-enabled Docker images for vision processing
   - Configure Uvicorn workers: `uvicorn app:app --workers 4`
   - Implement caching for reference vectors

2. **Security**:
   - Set up HTTPS/TLS
   - Implement authentication (API keys, OAuth)
   - Validate file types and sizes
   - Scan uploads for malware

3. **Monitoring**:
   - Add logging (structlog, loguru)
   - Implement metrics (Prometheus)
   - Set up health checks and alerts

4. **Scalability**:
   - Use load balancer for multiple instances
   - Implement async file processing
   - Consider cloud storage for uploads

---

## Project Strengths

1. **Multimodal Approach**: Combines complementary techniques for robust classification
2. **Flexible Architecture**: Modular design allows easy component replacement
3. **French Language Support**: Specialized NLP for French documents
4. **Modern Stack**: FastAPI, Transformers, PyTorch - industry-standard tools
5. **Containerization Ready**: Docker support for easy deployment
6. **Extensible**: Clean interfaces for adding new classification methods

## Future Enhancements

1. **Enhanced Evaluation**: Complete implementation of evaluation module
2. **Async Processing**: Background task queue for large batches
3. **Multi-language Support**: Extend NLP to other languages
4. **Fine-tuning**: Train ViT on domain-specific data
5. **Advanced Fusion**: Machine learning-based fusion (meta-classifier)
6. **API Enhancements**: Batch upload, progress tracking, webhooks
7. **Database Integration**: Store results, user management
8. **UI Dashboard**: Web interface for non-technical users

---

## Conclusion

This document classification project represents a sophisticated multimodal AI system that leverages both natural language processing and computer vision to accurately classify banking documents. The architecture is clean, modular, and production-ready, with clear separation of concerns and extensive configuration options.

The system successfully combines:
- **Text analysis** for semantic understanding
- **Visual analysis** for layout and formatting patterns
- **Weighted fusion** for optimal decision-making

This makes it particularly effective for real-world document classification tasks where documents may vary in format, quality, and structure.
