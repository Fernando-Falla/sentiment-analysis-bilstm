import re
import nltk
from nltk.corpus import stopwords
from functools import lru_cache

# Regex bilingüe optimizado
URL_PATTERN = re.compile(r"http\S+|www\S+|\S+@\S+")
CHAR_PATTERN = re.compile(r"[^a-záéíóúñüâêîôûàèìòùçãõ\s]") 
MULTISPACE_PATTERN = re.compile(r"\s+")

try:
    # Carga de stop words bilingües
    STOP_WORDS = set(stopwords.words('spanish')).union(set(stopwords.words('portuguese')))
except LookupError:
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('spanish')).union(set(stopwords.words('portuguese')))

@lru_cache(maxsize=1024)
def clean_text_optimized(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    
    text = text.lower()
    text = URL_PATTERN.sub("", text)
    text = CHAR_PATTERN.sub(" ", text) 
    text = MULTISPACE_PATTERN.sub(" ", text).strip()
    
    words = text.split()
    cleaned = [w for w in words if w not in STOP_WORDS]
    
    return " ".join(cleaned)
