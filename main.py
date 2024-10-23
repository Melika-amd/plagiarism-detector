import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


nlp = spacy.load("en_core_web_sm")


app = FastAPI()

class Document(BaseModel):
    text: str

class ComparisonRequest(BaseModel):
    original: str
    submission: str

def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def detect_patchwriting(original, submission):

    preprocessed_original = preprocess_text(original)
    preprocessed_submission = preprocess_text(submission)
    

    similarity = calculate_similarity(preprocessed_original, preprocessed_submission)
    

    original_sentences = [sent.text for sent in nlp(original).sents]
    submission_sentences = [sent.text for sent in nlp(submission).sents]
    

    similar_sentences = []
    for i, sub_sent in enumerate(submission_sentences):
        for j, orig_sent in enumerate(original_sentences):
            sent_similarity = calculate_similarity(preprocess_text(sub_sent), preprocess_text(orig_sent))
            if sent_similarity > 0.8:  
                similar_sentences.append({
                    "submission_sentence": sub_sent,
                    "original_sentence": orig_sent,
                    "similarity": sent_similarity
                })
    
    return {
        "overall_similarity": similarity,
        "similar_sentences": similar_sentences,
        "patchwriting_detected": len(similar_sentences) > 0 or similarity > 0.5
    }

@app.post("/check_plagiarism")
async def check_plagiarism(request: ComparisonRequest):
    try:
        result = detect_patchwriting(request.original, request.submission)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)