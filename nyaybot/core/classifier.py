import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crime_classifier.pkl")

_pipeline = None

def load_model():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        with open(MODEL_PATH, "rb") as f:
            _pipeline = pickle.load(f)


VALID_CASES = ["theft", "scam", "consumer", "cybercrime", "harassment"]
ML_CONFIDENCE_THRESHOLD = 0.50


def classify(user_input: str) -> dict:
    load_model()

    if not user_input or not user_input.strip():
        return {"crime_type": "unknown", "confidence": 0.0, "source": "none"}

    proba = _pipeline.predict_proba([user_input])[0]
    confidence = float(max(proba))
    predicted = _pipeline.predict([user_input])[0]

    if confidence >= ML_CONFIDENCE_THRESHOLD:
        return {
            "crime_type": predicted,
            "confidence": round(confidence, 2),
            "source": "ml"
        }
    else:
        return {
            "crime_type": "uncertain",
            "confidence": round(confidence, 2),
            "source": "llm_needed"
        }
        