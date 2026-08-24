import re
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer

# Bypass SSL verification for macOS NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK resources silently (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    """
    Context-Aware Text Preprocessor for Financial Phishing Detection.
    Strictly removes URLs (to prove they aren't relied upon) and focuses on semantic content.
    """
    def __init__(self, model_name="distilbert-base-uncased"):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Using DistilBERT tokenizer for contextual embedding later
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text by removing URLs, special characters, and extra spaces.
        """
        # 1. Handle URLs (Replace with the word 'link' to retain semantic context without relying on domain reputation)
        text = re.sub(r'http\S+|www.\S+', ' link ', text, flags=re.IGNORECASE)
        # 2. Handle email addresses (Replace with 'email')
        text = re.sub(r'\S+@\S+', ' email ', text)
        # 3. Keep alphanumeric and basic punctuation, lowercased
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text).lower()
        # 4. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize_text(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Applies lemmatization and optional stop-word removal.
        Note: Stop-word removal is often disabled for BERT-like models to preserve context.
        """
        words = text.split()
        if remove_stopwords:
            words = [w for w in words if w not in self.stop_words]
        
        words = [self.lemmatizer.lemmatize(w) for w in words]
        return ' '.join(words)

    def tokenize_for_bert(self, texts: list, max_length: int = 128):
        """
        Tokenizes text for DistilBERT/BERT consumption.
        Returns input_ids and attention_mask tensors.
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
