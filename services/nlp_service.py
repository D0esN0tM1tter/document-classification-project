import os
import re
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import pandas as pd
from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data (run once)
try:
    # Attempt to download the 'punkt' tokenizer data for NLTK
    nltk.download('punkt', quiet=True)
except Exception as e:
    # Raise a clear error if download fails
    raise RuntimeError("Failed to download NLTK 'punkt' data. Please check your internet connection or NLTK setup.") from e


class NLPModule :

    def __init__(self, keywords_path='data/keywords.txt', keyword_threshold=5):
        """
        Initialize NLP module
        
        Args:
            keywords_path: Path to keyword dictionary file
            keyword_threshold: Minimum number of keywords to classify as transaction
        """
        self.keywords = self.load_keywords(keywords_path)
        self.keyword_threshold = keyword_threshold
        self.stemmer = FrenchStemmer()
    
    def load_keywords(self, keywords_path):
        """Load keywords from file"""

        if os.path.exists(keywords_path):
            with open(keywords_path, 'r', encoding='utf-8') as f:
                keywords = [line.strip().lower() for line in f if line.strip()]
            print(f"Loaded {len(keywords)} keywords from {keywords_path}")
            return keywords
        
        else:
            print(f"Warning: Keywords file not found at {keywords_path}")
            return []
        

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF using PyPDF2 first, fall back to OCR if needed
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            extracted_text: String containing document text
        """
        text = ""
        
        try:
            # Method 1: Try PyPDF2 for text-based PDFs
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # If very little text extracted, try OCR
            if len(text.strip()) < 50:
                print(f"Little text found with PyPDF2, trying OCR for {pdf_path}")
                text = self.extract_text_with_ocr(pdf_path)
                
        except Exception as e:
            print(f"Error with PyPDF2 for {pdf_path}: {e}")
            print("Falling back to OCR...")
            text = self.extract_text_with_ocr(pdf_path)
        
        return text
    

    def preprocess_text(self, text):
        """
        Preprocess text: lowercase, tokenize, stem
        
        Args:
            text: Raw text string
            
        Returns:
            processed_tokens: List of processed tokens
        """
        # Lowercase
        text = text.lower()
        
        # Remove special characters but keep letters and numbers
        text = re.sub(r'[^a-zàâäçéèêëïîôùûüÿæœ0-9\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text, language='french')
        
        # Stem tokens
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        
        return stemmed_tokens
    

    def extract_text_with_ocr(self, pdf_path):
        """
        Extract text using OCR (Tesseract)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            extracted_text: String containing OCR'd text
        """
        text = ""
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            
            # OCR each page
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang='fra')
                text += page_text + "\n"
                
        except Exception as e:
            print(f"Error with OCR for {pdf_path}: {e}")
        
        return text
    
    def count_keyword_matches(self, text):
        """
        Count how many keywords appear in the text
        
        Args:
            text: Document text
            
        Returns:
            count: Number of keyword matches
            matched_keywords: List of keywords that were found
        """
        # Preprocess text
        tokens = self.preprocess_text(text)
        
        # Stem keywords
        stemmed_keywords = [self.stemmer.stem(kw) for kw in self.keywords]
        
        # Count matches
        matched = []
        for i, stemmed_kw in enumerate(stemmed_keywords):
            if stemmed_kw in tokens:
                matched.append(self.keywords[i])  # Store original keyword
        
        return len(matched), matched

    def predict(self, text):
        """
        Predict if document is a transaction based on keyword matching
        Args:
            text: Document text
        Returns:
            prediction: 'transaction' or 'non-transaction'
            count: keyword match count
            matched: list of matched keywords
        """
        count, matched = self.count_keyword_matches(text)
        if count >= self.keyword_threshold:
            return 'transaction', count, matched
        else:
            return 'non-transaction', count, matched
        


    def process_document(self, pdf_path):
        """
        Full pipeline: extract text, preprocess, predict, and normalize score.
        Args:
            pdf_path: Path to PDF file
        Returns:
            result: dict with keys 'prediction', 'score', 'matched_keywords', 'text', 'raw_count', 'max_possible'
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        # Predict using extracted text
        prediction, count, matched = self.predict(text)
        # Normalize score to [0, 1] based on number of keywords matched
        max_possible = len(self.keywords) if self.keywords else 1
        score = count / max_possible if max_possible > 0 else 0.0
        return {
            'prediction': prediction,
            'score': score,
            'matched_keywords': matched,
            'raw_count': count,
            'max_possible': max_possible
        }



