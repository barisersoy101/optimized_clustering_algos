import pandas as pd
import re
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.data import find
from sentence_transformers import SentenceTransformer
from langdetect import detect, LangDetectException
from spacy.cli import download as spacy_download

nltk_stopwords = set()

def ensure_nltk_resources():
    try:
        find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    global nltk_stopwords
    nltk_stopwords = set(stopwords.words('english'))

def ensure_spacy_model(model_name):
    try:
        spacy.load(model_name)
    except OSError:
        spacy_download(model_name)

def initialize_models(config):
    """Initialize models based on configuration."""
    ensure_nltk_resources()
    ensure_spacy_model(config['models']['spacy_model'])
    spacy_model = spacy.load(config['models']['spacy_model'])
    sbert_model = SentenceTransformer(config['models']['sbert_model'])
    return spacy_model, sbert_model

def normalize_text(text):
    """Normalize the text by cleaning and standardizing it."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', 'url', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s*\n\s*', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.\s*\.', '.', text)
    text = text.strip().strip(".") + "."
    return text

def detect_language(text):
    """Detect the language of the given text."""
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'

def split_sentences(text, spacy_model):
    """Split text into sentences using spaCy."""
    if not isinstance(text, str) or not text.strip():
        return []
    doc = spacy_model(text)
    return [sent.text.strip() for sent in doc.sents]

def filter_sentences(sentences, custom_keywords):
    """Filter out sentences containing custom keywords."""
    return [s for s in sentences if not any(word in s.lower() for word in custom_keywords)]

def get_sbert_embedding(text, spacy_model, sbert_model, custom_keywords):
    """Generate SBERT embedding for the filtered sentences."""
    sentences = split_sentences(text, spacy_model)
    filtered_sentences = filter_sentences(sentences, custom_keywords)
    if not filtered_sentences:
        return np.zeros(768)
    embeddings = sbert_model.encode(filtered_sentences)
    return np.mean(embeddings, axis=0)

def preprocess_text_column(df, column_name, config):
    """Preprocess the DataFrame for text clustering."""
    spacy_model, sbert_model = initialize_models(config)
    custom_keywords = config['custom_keywords']

    # Detect and filter non-English texts
    df['language'] = df[column_name].apply(detect_language)
    df = df[df['language'] == 'en']

    # Normalize, clean, and filter the text
    df['cleaned_text'] = df[column_name].apply(normalize_text)
    df['cleaned_text'] = df['cleaned_text'].apply(
        lambda x: " ".join(filter_sentences(split_sentences(x, spacy_model), custom_keywords))
    )

    # Filter out empty texts after cleaning
    df = df[df['cleaned_text'].str.strip() != ""]

    # Generate SBERT embeddings
    df['sbert_embeddings'] = df['cleaned_text'].apply(
        lambda x: get_sbert_embedding(x, spacy_model, sbert_model, custom_keywords)
    )

    # Filter out invalid embeddings
    df = df[df['sbert_embeddings'].apply(lambda x: isinstance(x, np.ndarray) and x.shape == (768,))]

    # Drop the original 'descriptions' column after preprocessing
    df = df.drop(columns=[column_name], errors='ignore')

    return df
