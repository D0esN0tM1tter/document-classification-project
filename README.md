# Document Classification Project

This project is a multimodal document classification API built with FastAPI. It combines Natural Language Processing (NLP) and Vision (image-based) techniques to classify PDF documents using both text and visual features.

## Features

- **Multimodal Classification:** Uses both text (NLP) and image (Vision Transformer) analysis for robust document classification.

- **REST API:** Easily upload and classify documents via HTTP endpoints.

- **Configurable:** Paths and model settings can be customized via environment variables.

## How It Works

1. **Upload a PDF document** using the `/classify-document` endpoint.
2. The system extracts text and images from the PDF.
3. **NLP Module:** Scans for keywords and uses stemming/tokenization to analyze text.
4. **Vision Module:** Uses a Vision Transformer (ViT) model to analyze document images and compare them to reference documents.
5. **Fusion:** Combines NLP and Vision results for a final prediction.

## Project Structure

```
app.py                  # FastAPI app and endpoints
schemas/                # Response models (Pydantic)
services/               # Core logic for NLP, Vision, and Fusion
data/                   # Keywords, reference docs, uploads
requirements.txt        # Python dependencies
Dockerfile              # Containerization support
tests/                  # Unit tests for modules
```

## Getting Started

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- (Optional) Docker

### Installation

1. **Clone the repository:**
	```
	git clone <your-repo-url>
	cd document-classfication-project
	```

2. **Install dependencies:**
	```
	pip install -r requirements.txt
	```

3. **Download NLTK data (first run only):**
	The app will attempt to download required NLTK data automatically.

### Running the API

Start the FastAPI server:
```
uvicorn app:app --port 8000 --reload
```

### API Usage

- **POST /classify-document**
  - Upload a PDF file to get its classification.
  - Response includes prediction, scores, and details from both NLP and Vision modules.

### Configuration

You can set environment variables to customize behavior:
- `NLP_KEYWORDS_PATH` (default: `data/keywords.txt`)
- `VISION_MODEL_IDENTIFIER` (default: `google/vit-base-patch16-224-in21k`)
- `VISION_REFERENCES` (default: `data/references`)
- `UPLOADS_PATH` (default: `data/uploads`)

