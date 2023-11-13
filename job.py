
import streamlit as st
import os
import sympy
import spacy
import pdfplumber
import pandas as pd
import plotly.express as px
import textacy
from langdetect import detect
from textacy import extract
import textdistance as td
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from pdf2image import convert_from_path
import pytesseract
import cv2
from plotly.subplots import make_subplots
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from nltk.corpus import stopwords

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

nltk.download('stopwords')
english_stopwords = set(stopwords.words('english'))
german_stopwords = set(stopwords.words('german'))

def get_common_terms(text1, text2, language):
    terms_text1 = set(text1)
    terms_text2 = set(text2)
    
    if language == 'en':
        terms_text1 = terms_text1 - english_stopwords
        terms_text2 = terms_text2 - english_stopwords
    elif language == 'de':
        terms_text1 = terms_text1 - german_stopwords
        terms_text2 = terms_text2 - german_stopwords

    common_terms = terms_text1.intersection(terms_text2)
    return list(common_terms)


    
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def pdf_dil_tespit(dosya_yolu_or_bytesio):
    with pdfplumber.open(dosya_yolu_or_bytesio) as pdf:
         metin = ''.join(page.extract_text() for page in pdf.pages)
    dil = detect(metin)
    return dil

def uyari_ver(is_ilani_metni, ozgecmis_klasoru):
    is_ilani_dili = detect(is_ilani_metni)
    ozgecmis_dosyalari = [os.path.join(ozgecmis_klasoru, dosya) for dosya in os.listdir(ozgecmis_klasoru) if dosya.endswith('.pdf')]
    
    farkli_dil_ozgecmisler = []
    uygun_ozgecmisler = []

    for ozgecmis in ozgecmis_dosyalari:
        ozgecmis_dili = pdf_dil_tespit(ozgecmis)
        if is_ilani_dili != ozgecmis_dili:
            st.warning(f"The resume '{os.path.basename(ozgecmis)}' is not in the same language as the job posting. Please upload it in the correct language to benefit from evaluation and extra services.")
            farkli_dil_ozgecmisler.append(os.path.basename(ozgecmis))
        else:
            uygun_ozgecmisler.append(ozgecmis)
    
    return uygun_ozgecmisler

nlp_english = spacy.load("en_core_web_lg")

def extract_terms_english(text):
    doc = nlp_english(text)
    return [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ', 'PROPN']]

nlp_german = spacy.load("de_core_news_lg")

def extract_terms_german(text):
    doc = nlp_german(text)
    return [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ', 'PROPN']]

def plot_word_cloud(terms, title):
    wordcloud = WordCloud(width=800, height=400, background_color='snow').generate(" ".join(terms))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=10)
    st.pyplot(plt.gcf())

def plot_pie_chart(data_frame, labels_column, values_column, title):
    if data_frame.empty or labels_column not in data_frame.columns or values_column not in data_frame.columns:
        fig = go.Figure()
        fig.add_annotation(text="The data set is empty or has missing columns.", xref="paper", yref="paper", showarrow=False, font=dict(size=15))
    else:
        fig = px.pie(data_frame, names=labels_column, values=values_column, title=title)
    return fig

def plot_bar_chart(data_frame, x, y, title):
    if data_frame.empty or x not in data_frame.columns or y not in data_frame.columns:
        fig = go.Figure()
        fig.add_annotation(text="The data set is empty or has missing columns.", xref="paper", yref="paper", showarrow=False, font=dict(size=15))
    else:
        data_frame['color'] = range(1, len(data_frame) + 1)
        fig = px.bar(data_frame, x=x, y=y, title=title, color='color', color_continuous_scale=px.colors.qualitative.Set1)
        fig.update_layout(showlegend=False)
    return fig

def plot_side_by_side(pie_df, bar_df):
    fig1 = plot_pie_chart(pie_df, 'Resume', 'Similarity Score', 'Similarity Scores According to Job Postings')
    fig2 = plot_bar_chart(bar_df, 'Resume', 'Similarity Score', 'Similarity Scores')

    subplot_fig = make_subplots(rows=1, cols=2, 
                                specs=[[{'type': 'domain'}, {'type': 'xy'}]], 
                                subplot_titles=('Similarity Scores According to Job Postings', 'Similarity Scores'))

    for trace in fig1.data:
        trace.showlegend = False
        subplot_fig.add_trace(trace, row=1, col=1)

    for trace in fig2.data:
        subplot_fig.add_trace(trace, row=1, col=2)

    subplot_fig.update_layout(margin=dict(t=50, b=50, r=50, l=50), 
                              bargap=0.1, showlegend=False)
    
    return subplot_fig

def plot_line_chart(data_frame, x, title):
    fig = px.line(data_frame, x=x, y=data_frame.columns[1:], title=title, markers=True)
    st.plotly_chart(fig)
def metrics_definitions():
    st.title("Metrics Definitions")
    
    st.write("""
    ### Jaccard Similarity:
    - **Explanation:** Jaccard similarity measure divides the intersection of two sets by their union.
    - **Benefits:** Useful for comparing the similarity between two sets. Ideal for situations where the presence or absence of a term is more significant than its frequency.
        """)
    st.write("Mathematical Definition: J(A, B) = |A ∩ B| / |A ∪ B|")
    
    st.write("""
    ### Sorensen-Dice Similarity:
    - **Explanation:** Sorensen-Dice similarity measure divides the intersection of two sets by the total number of elements in the two sets.
    - **Benefits:** This metric provides a balance between precision and recall in its evaluation.
    """)
    st.write("Mathematical Definition: S(A, B) = 2 * |A ∩ B| / (|A| + |B|)")

    st.write("""
    ### Cosine Similarity:
    - **Explanation:** Cosine similarity measure is calculated using the cosine of the angles between two vectors.
    - **Benefits:** Cosine similarity is particularly used in positive space, where the outcome is neatly bounded in [0,1]. It is a measure of similarity between two non-zero vectors.
    """)
    st.write("Mathematical Definition: Cosine(A, B) = (A · B) / (||A|| * ||B||)")

    st.write("""
    ### Overlap Similarity:
    - **Explanation:** Overlap similarity measure divides the intersection of two sets by the number of elements in the smaller set.
    - **Benefits:** Useful for situations where you want to find out how much of the smaller set is represented in the larger set.
    """)
    st.write("Mathematical Definition: O(A, B) = |A ∩ B| / min(|A|, |B|)")
    
    st.write("""
    ### Word Mover's Distance (Word2Vec Similarity):
    - A measure of text similarity that considers the semantic meaning of words.
    - Utilizes word embeddings, like those from Word2Vec, to represent words in high-dimensional space.
    - Calculates the minimum "distance" or "cost" required to align or "move" the words in one text to match those in another.
    - Lower scores indicate greater similarity in meaning between texts.
    - **Benefits:** It captures semantic meaning of words in texts, making it ideal for comparing texts with different structures but similar meaning.
    """)

    st.write("""
    ### Cosine Similarity (Sentence Transformers):
    - A metric to measure the similarity between two vectors, commonly used with text data.
    - Uses sentence embeddings from models like Sentence Transformers to represent entire sentences or documents as vectors.
    - Computes the cosine of the angle between two vectors; scores close to 1 mean high similarity, while those close to 0 mean low similarity.
    - Efficient for capturing the overall semantic meaning of longer texts.
    - **Benefits:** Efficient for capturing the overall semantic meaning of longer texts.
    """)


    
def run_english_version():
    style = """
    <style>
    body {
        background-color: #E6E6FA; 
        color: #000000;
    }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    resume_folder = st.sidebar.text_input("Folder path where resumes are saved:").strip()
    JD_option = st.selectbox("How will the job description be given?", ["Text", "PDF"])
    JD = None
    if JD_option == "Text":
        JD = st.text_area("Enter job description:")
    else:
        JD_file = st.file_uploader("Upload PDF file for job description", type="pdf")
        if JD_file is not None:
            JD = extract_text_from_pdf(BytesIO(JD_file.read()))

    if JD:
        suitable_resumes = uyari_ver(JD, resume_folder)
        st.subheader("Job Posting Text:")
        st.write(JD)
        
        job_description_terms = extract_terms_english(JD)
        #st.subheader("Keywords Extracted:")
        #st.write(", ".join(job_description_terms))
        
        plot_word_cloud(job_description_terms, "Job Posting - WordCloud")
        
        results = []
        advanced_similarity_scores = []
        
        transformers_model_name = "bert-base-nli-mean-tokens"
        sentence_transformers_model = SentenceTransformer(transformers_model_name, device="cuda" if torch.cuda.is_available() else "cpu")

        for file in suitable_resumes:
            resume_data = extract_text_from_pdf(file)
            
            # İngilizce NLP işlemi için extract_terms_english işlevi kullanılıyor
            resume_terms = extract_terms_english(resume_data)
            
            common_terms = set(job_description_terms).intersection(resume_terms)
            
            resume_embedding = sentence_transformers_model.encode([resume_data])
            job_posting_embedding = sentence_transformers_model.encode([JD])
            cosine_similarity_score = util.pytorch_cos_sim(resume_embedding, job_posting_embedding)[0][0].item()
            
            resume_doc = textacy.make_spacy_doc(resume_data, lang="en_core_web_lg")
            jd_doc = textacy.make_spacy_doc(JD, lang="en_core_web_lg")
            resume_keyterms = {term for term, _ in extract.keyterms.textrank(resume_doc, normalize="lemma", topn=100)}
            jd_keyterms = {term for term, _ in extract.keyterms.textrank(jd_doc, normalize="lemma", topn=100)}
            
            j = td.jaccard.similarity(resume_keyterms, jd_keyterms) * 100
            s = td.sorensen_dice.similarity(resume_keyterms, jd_keyterms) * 100
            c = td.cosine.similarity(resume_keyterms, jd_keyterms) * 100
            o = td.overlap.normalized_similarity(resume_keyterms, jd_keyterms) * 100
            
            model_word2vec = Word2Vec([job_description_terms, resume_terms], vector_size=100, window=5, min_count=1, workers=4)

            instance = WmdSimilarity([job_description_terms], model_word2vec.wv, num_best=1)
            similarity_wmd = instance[resume_terms][0][1]
            
            similarity_score = (j + s + c + o + cosine_similarity_score) / 5
            
            advanced_similarity_scores.append({
                'Resume': os.path.basename(file),
                'Jaccard': j,
                'Sorensen-Dice': s,
                'Cosine': c,
                'Overlap': o,
                'Word Mover\'s Distance(Semantic Sim.)': similarity_wmd,
                'Cosine Similarity (Sentence Transformers)': cosine_similarity_score
            })
            
            results.append({
                'Resume': os.path.basename(file),
                'Common Terms': ', '.join(common_terms),
                'Number of Common Terms': len(common_terms),
                'Similarity Score': similarity_score
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Number of Common Terms', ascending=False)
        st.table(results_df)

        combined_figure = plot_side_by_side(results_df, results_df)
        st.plotly_chart(combined_figure)
        
        advanced_similarity_df = pd.DataFrame(advanced_similarity_scores)
        advanced_similarity_df = advanced_similarity_df.sort_values(by='Jaccard', ascending=False)
        st.subheader("Text Similarity with Advanced Techniques and Measurements")
        st.table(advanced_similarity_df)
        
        plot_line_chart(advanced_similarity_df, 'Resume', 'Different Similarity Scores')
        
        max_score_index = results_df['Similarity Score'].idxmax()
        max_score_resume = results_df.loc[max_score_index, 'Resume']
        st.markdown(f"### Highest Similarity Score: {max_score_resume}")

        max_score_file = os.path.join(resume_folder, max_score_resume)
        max_score_resume_content = extract_text_from_pdf(max_score_file)
        expander = st.expander("View Resume Content")
        for paragraph in max_score_resume_content.split('\n\n'):
            expander.markdown(f"> {paragraph}")



def run_german_version():
    style = """
    <style>
    body {
        background-color: #F5F5DC; 
        color: #000000;
    }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    resume_folder = st.sidebar.text_input("Folder path where CVs are stored:").strip()
    JD_option = st.selectbox("How will the job description be provided?", ["Text", "PDF"])
    JD = None
    if JD_option == "Text":
        JD = st.text_area("Enter job description:")
    else:
        JD_file = st.file_uploader("Upload PDF file for job description", type="pdf")
        if JD_file is not None:
            JD = extract_text_from_pdf(BytesIO(JD_file.read()))

    if JD:
        suitable_resumes = uyari_ver(JD, resume_folder)
        st.subheader("Job Posting Text:")
        st.write(JD)
        
        job_description_terms = extract_terms_german(JD)
        #st.subheader("Extracted Keywords:")
        #st.write(", ".join(job_description_terms))
        
        plot_word_cloud(job_description_terms, "Job Posting - WordCloud")

        results = []
        advanced_similarity_scores = []
        
        transformers_model_name = "T-Systems-onsite/german-roberta-sentence-transformer-v2"
        sentence_transformers_model = SentenceTransformer(transformers_model_name, device="cuda" if torch.cuda.is_available() else "cpu")

        for file in suitable_resumes:
            resume_data = extract_text_from_pdf(file)
            
            resume_terms = extract_terms_german(resume_data)
            common_terms = set(job_description_terms).intersection(resume_terms)
            
            resume_embedding = sentence_transformers_model.encode([resume_data])
            job_posting_embedding = sentence_transformers_model.encode([JD])
            cosine_similarity_score = util.pytorch_cos_sim(resume_embedding, job_posting_embedding)[0][0].item()
            
            resume_doc = textacy.make_spacy_doc(resume_data, lang="de_core_news_lg")
            jd_doc = textacy.make_spacy_doc(JD, lang="de_core_news_lg")
            resume_keyterms = {term for term, _ in extract.keyterms.textrank(resume_doc, normalize="lemma", topn=100)}
            jd_keyterms = {term for term, _ in extract.keyterms.textrank(jd_doc, normalize="lemma", topn=100)}
            
            j = td.jaccard.similarity(resume_keyterms, jd_keyterms) * 100
            s = td.sorensen_dice.similarity(resume_keyterms, jd_keyterms) * 100
            c = td.cosine.similarity(resume_keyterms, jd_keyterms) * 100
            o = td.overlap.normalized_similarity(resume_keyterms, jd_keyterms) * 100
            
            model_word2vec = Word2Vec([job_description_terms, resume_terms], vector_size=100, window=5, min_count=1, workers=4)
            instance = WmdSimilarity([job_description_terms], model_word2vec.wv, num_best=1)
            similarity_wmd = instance[resume_terms][0][1]
            
            similarity_score = (j + s + c + o + cosine_similarity_score) / 5
            
            advanced_similarity_scores.append({
                'Resume': os.path.basename(file),
                'Jaccard': j,
                'Sorensen-Dice': s,
                'Cosine': c,
                'Overlap': o,
                'Word Mover\'s Distance(Semantic Sim.)': similarity_wmd,
                'Cosine Similarity (Sentence Transformers)': cosine_similarity_score
            })
            
            results.append({
                'Resume': os.path.basename(file),
                'Common Terms': ', '.join(common_terms),
                'Number of Common Terms': len(common_terms),
                'Similarity Score': similarity_score
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Number of Common Terms', ascending=False)
        st.table(results_df)

        combined_figure = plot_side_by_side(results_df, results_df)
        st.plotly_chart(combined_figure)
        
        advanced_similarity_df = pd.DataFrame(advanced_similarity_scores)
        advanced_similarity_df = advanced_similarity_df.sort_values(by='Jaccard', ascending=False)
        st.subheader("Text Similarity with Advanced Techniques and Measurements")
        st.table(advanced_similarity_df)
        
        plot_line_chart(advanced_similarity_df, 'Resume', 'Different Similarity Scores')
        
        max_score_index = results_df['Similarity Score'].idxmax()
        max_score_resume = results_df.loc[max_score_index, 'Resume']
        st.markdown(f"### Highest Similarity Score: {max_score_resume}")

        max_score_file = os.path.join(resume_folder, max_score_resume)
        max_score_resume_content = extract_text_from_pdf(max_score_file)
        expander = st.expander("View Resume Content")
        for paragraph in max_score_resume_content.split('\n\n'):
            expander.markdown(f"> {paragraph}")


def main():
    st.title("Job Advertisement-Resume Monitoring and Evaluation System")
   

    page = st.sidebar.selectbox("Choose a page", ["Home", "Metrics Definitions"])
    
    if page == "Home":
        language_option = st.sidebar.selectbox("Please select a language:", ["English", "German"])
        
        if language_option == "English":
            run_english_version()
        elif language_option == "German":
            run_german_version()
    elif page == "Metrics Definitions":
        metrics_definitions()

if __name__ == "__main__":
    main()





