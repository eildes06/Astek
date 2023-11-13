
import streamlit as st
import pdfplumber
from langdetect import detect
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from wordcloud import WordCloud
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# İlk defa nltk ile stopwords kullanıyorsanız, bu veriyi indirmeniz gerekebilir
nltk.download('stopwords')

english_stopwords = set(stopwords.words('english'))
german_stopwords = set(stopwords.words('german'))

def get_common_terms(text1, text2, language):
    if language == "en":
        terms1 = set(text1.split()) - english_stopwords
        terms2 = set(text2.split()) - english_stopwords
    elif language == "de":
        terms1 = set(text1.split()) - german_stopwords
        terms2 = set(text2.split()) - german_stopwords
    else:
        terms1 = set(text1.split())
        terms2 = set(text2.split())
    return list(terms1.intersection(terms2))


def detect_language(text):
    """Detect the language of the text"""
    return detect(text)

def detect_pdf_language(pdf_path_or_bytesio):
    with pdfplumber.open(pdf_path_or_bytesio) as pdf:
        text = ''.join(page.extract_text() for page in pdf.pages)
    return detect_language(text)

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text



def plot_word_cloud(terms, title):
    wordcloud = WordCloud(width=800, height=400, background_color='snow', scale=2).generate(" ".join(terms))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=10)
    st.pyplot(plt.gcf())

def evaluate_similarity(job_text, cv_text, tokenizer, model, sentence_transformers_model):
    
    # Dil tespiti
    language = detect_language(job_text)
    
    job_text = preprocess(job_text)
    cv_text = preprocess(cv_text)

    # Dil bilgisini get_common_terms fonksiyonuna iletiyoruz
    common_terms = get_common_terms(job_text, cv_text, language)
    common_terms_count = len(common_terms)
    common_terms_str = ', '.join(common_terms)

    # Compute Word Mover’s Distance
    job_terms = job_text.split()
    cv_terms = cv_text.split()
    model_word2vec = Word2Vec([job_terms, cv_terms], vector_size=100, window=5, min_count=1, workers=4)
    instance = WmdSimilarity([job_terms], model_word2vec.wv, num_best=1)
    similarity_wmd = instance[cv_terms][0][1]

    # Compute BERT-based Cosine Similarity
    job_tokens = tokenizer(job_text, return_tensors="pt", padding=True, truncation=True)
    cv_tokens = tokenizer(cv_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        job_embeddings = model(**job_tokens).last_hidden_state.mean(dim=1)
        cv_embeddings = model(**cv_tokens).last_hidden_state.mean(dim=1)
    similarity_cosine = cosine_similarity(job_embeddings.cpu(), cv_embeddings.cpu())[0][0]

    # Convert resume and job posting texts to embeddings using Sentence Transformers
    resume_embedding = sentence_transformers_model.encode(cv_text)
    job_posting_embedding = sentence_transformers_model.encode(job_text)
    cosine_similarity_score = util.pytorch_cos_sim([resume_embedding], [job_posting_embedding]).item()

    # Weighting
    weight_cosine = 0.30
    weight_wmd = 0.35
    weight_cosine_similarity = 0.35
    final_similarity = (weight_cosine * similarity_cosine + weight_wmd * similarity_wmd + weight_cosine_similarity * cosine_similarity_score) / (weight_cosine + weight_wmd + weight_cosine_similarity)

    results = {
        "Similarity Measurements": [
            "Weighted Scoring (Final Similarity)",
            "Bert-Cosine Similarity",
            "Sentence Transformers-Cosine Similarity",
            "Word Mover's Distance",
            "Common Words",
            "Common Words Count"
        ],
        "Results": [
            final_similarity,
            similarity_cosine,
            cosine_similarity_score,
            similarity_wmd,
            common_terms_str,
            common_terms_count
        ]
    }

    return pd.DataFrame(results), common_terms





def evaluate_in_english(job_text, cv_text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased")
    sentence_transformers_model = SentenceTransformer("bert-base-nli-mean-tokens", device="cuda" if torch.cuda.is_available() else "cpu")
    return evaluate_similarity(job_text, cv_text, tokenizer, model, sentence_transformers_model)

def evaluate_in_german(job_text, cv_text):
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
    model = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")
    sentence_transformers_model = SentenceTransformer("T-Systems-onsite/german-roberta-sentence-transformer-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    return evaluate_similarity(job_text, cv_text, tokenizer, model, sentence_transformers_model)


# Streamlit Application
st.title("Job Matching App")

language_option = st.selectbox("Choose a language for evaluation:", ["English", "German"])

uploaded_file_job = st.file_uploader(f"Choose a Job Description PDF file ({language_option})", type="pdf")
uploaded_file_cv = st.file_uploader(f"Choose a CV PDF file ({language_option})", type="pdf")

if st.button('Compare'): # Buton, dosya yükleyicilerin hemen altında
    
    if uploaded_file_job and uploaded_file_cv:
        job_text = extract_text_from_pdf(uploaded_file_job)
        cv_text = extract_text_from_pdf(uploaded_file_cv)

        # Dil kontrolü
        job_language = detect_pdf_language(uploaded_file_job)
        cv_language = detect_pdf_language(uploaded_file_cv)
        
        if job_language != cv_language:
            st.warning("The CV and job description are not in the same language and cannot be compared.")
        else:
            if language_option == "English" and job_language == "en" and cv_language == "en":
                results_df, common_terms = evaluate_in_english(job_text, cv_text)
                st.table(results_df)
                plot_word_cloud(common_terms, "Common Words")
            elif language_option == "German" and job_language == "de" and cv_language == "de":
                results_df, common_terms = evaluate_in_german(job_text, cv_text)
                st.table(results_df)
                plot_word_cloud(common_terms, "Common Words")
            else:
                st.warning(f"The selected language {language_option} does not match the detected language(s) of the uploaded files.")

