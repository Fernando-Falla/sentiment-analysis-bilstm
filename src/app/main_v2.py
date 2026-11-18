import os
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.app.utils_v2 import clean_text_optimized

# Optimización para 1 OCPU Ampere
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = FastAPI(title="Sentiment API - G68 Final")

# Rutas estándar dentro del contenedor
BASE_MODELS_PATH = "/app/data/models"
MODEL_FILE = os.path.join(BASE_MODELS_PATH, "modelo_lstm_v2.keras")
TOKENIZER_FILE = os.path.join(BASE_MODELS_PATH, "tokenizer.json")
METADATA_FILE = os.path.join(BASE_MODELS_PATH, "metadata.json")

try:
    with open(TOKENIZER_FILE, 'r', encoding='utf-8') as f:
        tokenizer = tokenizer_from_json(json.load(f))
    model = tf.keras.models.load_model(MODEL_FILE)
    with open(METADATA_FILE, 'r') as f:
        meta = json.load(f)
        classes = {int(k): v for k, v in meta['labels'].items()}
        MAX_LEN = meta.get('max_len', 200)
    print("✅ Artefactos de IA cargados con éxito.")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")
    raise e

# Seguridad
API_KEY_SENTIMENT = os.getenv("API_KEY_SENTIMENT")
if not API_KEY_SENTIMENT:
    raise RuntimeError("Defina API_KEY_SENTIMENT en el entorno (ver .env.example).")
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def get_api_key(header_key: str = Depends(api_key_header)):
    if header_key != API_KEY_SENTIMENT:
        raise HTTPException(status_code=403, detail="API Key no válida")
    return header_key

class SentimentRequest(BaseModel):
    text: str

def get_top_features(text, original_prob, original_class_idx):
    words = text.split()
    if len(words) <= 1: return words
    variations = [" ".join(words[:i] + words[i+1:]) for i in range(len(words))]
    seqs = tokenizer.texts_to_sequences(variations)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    preds = model.predict(padded, verbose=0)
    impacts = [(words[i], original_prob - p[original_class_idx]) for i, p in enumerate(preds)]
    impacts.sort(key=lambda x: x[1], reverse=True)
    return [word for word, score in impacts[:3]]

@app.post("/predict/v2/sentiment", dependencies=[Depends(get_api_key)])
async def predict(request: SentimentRequest):
    cleaned = clean_text_optimized(request.text)
    if not cleaned:
        return {"prevision": "neutro", "probabilidad": 0.0, "top_features": []}

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prediction = model.predict(padded, verbose=0)
    
    idx = np.argmax(prediction[0])
    prob = float(np.max(prediction[0]))
    
    return {
        "prevision": classes[idx],
        "probabilidad": round(prob, 4),
        "texto_limpio": cleaned,
        "top_features": get_top_features(cleaned, prob, idx)
    }
