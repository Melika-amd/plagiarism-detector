import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict
import re
from collections import Counter
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

nlp = spacy.load("en_core_web_sm")

app = FastAPI(
    title="Plagiarism Detection API",
    description="API for detecting plagiarism and patchwriting using NLP techniques",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")

@app.get("/check")
async def check_endpoint():
    return RedirectResponse(url="/")

class Document(BaseModel):
    text: str = Field(..., min_length=10, description="Text content to analyze")

class ComparisonRequest(BaseModel):
    original: str = Field(..., min_length=10, description="Original text")
    submission: str = Field(..., min_length=10, description="Submitted text to check")

class SimilarityMatch(BaseModel):
    submission_sentence: str
    original_sentence: str
    similarity: float
    match_type: str

class PlagiarismResponse(BaseModel):
    overall_similarity: float
    similar_sentences: List[Dict]
    patchwriting_detected: bool
    text_statistics: Dict
    ngram_matches: List[Dict]
    similarity_metrics: Dict

def get_text_statistics(text: str) -> Dict:
    doc = nlp(text)
    word_tokens = [token for token in doc if not token.is_punct]
    return {
        "word_count": len(word_tokens),
        "sentence_count": len(list(doc.sents)),
        "average_word_length": np.mean([len(token.text) for token in word_tokens]),
        "unique_words": len(set([token.text.lower() for token in word_tokens])),
    }

def extract_ngrams(text: str, n: int) -> List[str]:
    words = [token.text.lower() for token in nlp(text) if not token.is_punct and not token.is_stop]
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def find_ngram_matches(original: str, submission: str, n: int = 3) -> List[Dict]:
    original_ngrams = extract_ngrams(original, n)
    submission_ngrams = extract_ngrams(submission, n)
    
    matches = []
    for i, sub_ngram in enumerate(submission_ngrams):
        if sub_ngram in original_ngrams:
            matches.append({
                "ngram": sub_ngram,
                "position": i,
                "length": n
            })
    return matches

def calculate_similarity_metrics(text1: str, text2: str) -> Dict:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
    
    return {
        "cosine_similarity": float(cosine_sim),
        "jaccard_similarity": float(jaccard)
    }

def preprocess_text(text: str) -> str:
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc 
              if not token.is_stop and not token.is_punct and len(token.text.strip()) > 1]
    return " ".join(tokens)

def detect_patchwriting(original: str, submission: str) -> Dict:
    if not original or not submission:
        raise ValueError("Both original and submission texts must be provided")
        
    try:
        preprocessed_original = preprocess_text(original)
        preprocessed_submission = preprocess_text(submission)
    except Exception as e:
        raise ValueError(f"Error processing text: {str(e)}")
    
    similarity_metrics = calculate_similarity_metrics(preprocessed_original, preprocessed_submission)
    
    original_sentences = [sent.text.strip() for sent in nlp(original).sents]
    submission_sentences = [sent.text.strip() for sent in nlp(submission).sents]
    
    similar_sentences = []
    for sub_sent in submission_sentences:
        for orig_sent in original_sentences:
            sent_similarity = calculate_similarity_metrics(
                preprocess_text(sub_sent), 
                preprocess_text(orig_sent)
            )["cosine_similarity"]
            
            if sent_similarity > 0.8:
                similar_sentences.append({
                    "submission_sentence": sub_sent,
                    "original_sentence": orig_sent,
                    "similarity": sent_similarity,
                    "match_type": "High Similarity"
                })
    
    ngram_matches = find_ngram_matches(original, submission)
    submission_stats = get_text_statistics(submission)
    
    return {
        "overall_similarity": similarity_metrics["cosine_similarity"],
        "similar_sentences": similar_sentences,
        "patchwriting_detected": len(similar_sentences) > 0 or similarity_metrics["cosine_similarity"] > 0.5,
        "text_statistics": submission_stats,
        "ngram_matches": ngram_matches,
        "similarity_metrics": similarity_metrics
    }

@app.post("/check_plagiarism", response_model=PlagiarismResponse)
async def check_plagiarism(request: ComparisonRequest):
    try:
        original = request.original.strip()
        submission = request.submission.strip()
        
        if len(original) < 10 or len(submission) < 10:
            raise HTTPException(
                status_code=400,
                detail="Both original and submission texts must be at least 10 characters long"
            )
        
        if len(original) > 50000 or len(submission) > 50000:
            raise HTTPException(
                status_code=400,
                detail="Text length exceeds maximum limit of 50,000 characters"
            )
        
        result = detect_patchwriting(original, submission)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)