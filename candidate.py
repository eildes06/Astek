


import streamlit as st
import os
import sympy
import spacy
import pdfplumber
import streamlit as st
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
from pdf2image import convert_from_path
import pytesseract
import textract
from sentence_transformers import SentenceTransformer, util
import torch

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def format_content(content):
    lines = content.split('\n')
    combined_lines = []
    current_line = ""

    for line in lines:
        if len(line) == 0:
            if len(current_line) > 0:
                combined_lines.append(current_line)
                current_line = ""
        else:
            if len(current_line) > 0:
                current_line += " " + line
            else:
                current_line = line

    if len(current_line) > 0:
        combined_lines.append(current_line)

    return "\n".join(combined_lines)

def extract_text_from_pdf_with_textract(pdf_path):
    try:
        text = textract.process(pdf_path)
        formatted_text = format_content(text.decode("utf-8"))
        return formatted_text
    except Exception as e:
        print(f"Error in extracting text from {pdf_path}: {str(e)}")
        return ""

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


def clean_content_tesseract(text):
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        if line.strip() != '':
            cleaned_lines.append(line.strip())

    paragraphs = []
    current_paragraph = []
    for line in cleaned_lines:
        if len(line.split()) <= 3:  # If the line has 3 or fewer words, consider it as a part of the same paragraph
            current_paragraph.append(line)
        else:
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
            paragraphs.append(line)
    if current_paragraph:  # Add any remaining content
        paragraphs.append(' '.join(current_paragraph))
    
    return '\n\n'.join(paragraphs)


def uyari_ver_is_ilanlari(resume, job_postings_folder):
    resume_dili = detect(resume)
    is_ilanlari_dosyalari = [os.path.join(job_postings_folder, dosya) for dosya in os.listdir(job_postings_folder) if dosya.endswith('.pdf')]
    
    farkli_dil_is_ilanlari = []
    uygun_is_ilanlari = []

    for is_ilani in is_ilanlari_dosyalari:
        is_ilani_dili = pdf_dil_tespit(is_ilani)
        if resume_dili != is_ilani_dili:
            st.warning(f"The job posting '{os.path.basename(is_ilani)}' is not in the same language as the resume. It won't be evaluated.")
            farkli_dil_is_ilanlari.append(os.path.basename(is_ilani))
        else:
            uygun_is_ilanlari.append(is_ilani)
    
    return uygun_is_ilanlari



nlp_english = spacy.load("en_core_web_lg")  # İNGİLİZCE SPACY LARGE DİL MODELİ


def extract_terms_english(text):
    doc = nlp_english(text)
    return [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ', 'PROPN']]


nlp_german = spacy.load("de_core_news_lg")  # ALMANCA SPACY LARGE DİL MODELİ


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

    
def run_english_version(resume, job_postings_folder):
    style = """
    <style>
    body {
    background-color: #E6E6FA; 
    color: #000000;
    }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    
    if resume:
        uygun_is_ilanlari = uyari_ver_is_ilanlari(resume, job_postings_folder)
        
        st.subheader("Resume Text:")
        st.write(resume)
        
        resume_terms = extract_terms_english(resume)
       # st.subheader("Keywords Extracted from Resume:")
       # st.write(", ".join(resume_terms))
        
        plot_word_cloud(resume_terms, "Resume - WordCloud")
        
        if st.button("Compare!", key="en_compare_button"): 
            if job_postings_folder and os.path.exists(job_postings_folder):
                uploaded_files = [os.path.join(job_postings_folder, f) for f in os.listdir(job_postings_folder) if f.endswith('.pdf')]
                if not uploaded_files:
                    st.error("Error: No PDF files found in the specified folder!")
                    return
                
                results = []
                advanced_similarity_scores = []
                
                transformers_model_name = "bert-base-nli-mean-tokens"  #BERT-ENGLISH(SENTENCE TRANSFORMER)
                sentence_transformers_model = SentenceTransformer(transformers_model_name, device="cuda" if torch.cuda.is_available() else "cpu")
                for file in uygun_is_ilanlari:
                    JD = extract_text_from_pdf(file)
                    job_description_terms = extract_terms_english(JD)
                    common_terms = set(job_description_terms).intersection(resume_terms)
                    
                    resume_embedding = sentence_transformers_model.encode([resume])
                    job_posting_embedding = sentence_transformers_model.encode([JD])
                    cosine_similarity_score = util.pytorch_cos_sim(resume_embedding, job_posting_embedding)[0][0].item()
                    
                    j = td.jaccard.similarity(resume_terms, job_description_terms) * 100
                    s = td.sorensen_dice.similarity(resume_terms, job_description_terms) * 100
                    c = td.cosine.similarity(resume_terms, job_description_terms) * 100
                    o = td.overlap.normalized_similarity(resume_terms, job_description_terms) * 100
                    
                    similarity_score = (j + s + c + o + cosine_similarity_score) / 5
                    
                    advanced_similarity_scores.append({
                        'Resume': os.path.basename(file),
                        'Jaccard': j,
                        'Sorensen-Dice': s,
                        'Cosine': c,
                        'Overlap': o,
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
                max_score_job_posting = results_df.loc[max_score_index, 'Resume']
                st.markdown(f"### Highest Similarity Score: {max_score_job_posting}")
                # Tesseract ile metni çıkar
                max_score_file = os.path.join(job_postings_folder, max_score_job_posting)

                pages = convert_from_path(max_score_file)
                max_score_job_posting_content = ""
                for page in pages:
                    max_score_job_posting_content += pytesseract.image_to_string(page)

                # İçeriği düzenle
                cleaned_job_posting_content = clean_content_tesseract(max_score_job_posting_content)
                expander = st.expander("View Job Posting Content")
                for paragraph in cleaned_job_posting_content.split('\n\n'):
                    expander.markdown(f"> {paragraph}")
            else:
                st.error("Error: The specified folder path cannot be reached. Please enter a valid folder path.")

def run_german_version(resume, job_postings_folder):
    style = """
    <style>
    body {
    background-color: #E6E6FA; 
    color: #000000;
    }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

    if resume:
        uygun_is_ilanlari = uyari_ver_is_ilanlari(resume, job_postings_folder)

        st.subheader("Resume Text:")
        st.write(resume)
        
        resume_terms = extract_terms_german(resume)
       # st.subheader("Keywords Extracted from Resume:")
       # st.write(", ".join(resume_terms))
        
        plot_word_cloud(resume_terms, "Resume - WordCloud")
        
        if st.button("Compare!"):
            if job_postings_folder and os.path.exists(job_postings_folder):
                uploaded_files = [os.path.join(job_postings_folder, f) for f in os.listdir(job_postings_folder) if f.endswith('.pdf')]
                if not uploaded_files:
                    st.error("Error: No PDF files found in the specified folder!")
                    return
                
                results = []
                advanced_similarity_scores = []
                
                transformers_model_name = "T-Systems-onsite/german-roberta-sentence-transformer-v2" # SENTENCE TRANSFORMERS ALMANCA MODELİ
                sentence_transformers_model = SentenceTransformer(transformers_model_name, device="cuda" if torch.cuda.is_available() else "cpu")
                for file in uygun_is_ilanlari:
                    JD = extract_text_from_pdf(file)
                    job_description_terms = extract_terms_english(JD)
                    common_terms = set(job_description_terms).intersection(resume_terms)
                    
                    resume_embedding = sentence_transformers_model.encode([resume])
                    job_posting_embedding = sentence_transformers_model.encode([JD])
                    cosine_similarity_score = util.pytorch_cos_sim(resume_embedding, job_posting_embedding)[0][0].item()
                    
                    j = td.jaccard.similarity(resume_terms, job_description_terms) * 100
                    s = td.sorensen_dice.similarity(resume_terms, job_description_terms) * 100
                    c = td.cosine.similarity(resume_terms, job_description_terms) * 100
                    o = td.overlap.normalized_similarity(resume_terms, job_description_terms) * 100
                    
                    similarity_score = (j + s + c + o + cosine_similarity_score) / 5
                    advanced_similarity_scores.append({
                        'Resume': os.path.basename(file),
                        'Jaccard': j,
                        'Sorensen-Dice': s,
                        'Cosine': c,
                        'Overlap': o,
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
                max_score_job_posting = results_df.loc[max_score_index, 'Resume']
                st.markdown(f"### Highest Similarity Score: {max_score_job_posting}")

                max_score_file = os.path.join(job_postings_folder, max_score_job_posting)
                

                 # Tesseract ile metni çıkar
                pages = convert_from_path(max_score_file)
                max_score_job_posting_content = ""
                for page in pages:
                    max_score_job_posting_content += pytesseract.image_to_string(page)

                # İçeriği düzenle
                cleaned_job_posting_content = clean_content_tesseract(max_score_job_posting_content)
                expander = st.expander("View Job Posting Content")
                for paragraph in cleaned_job_posting_content.split('\n\n'):
                    expander.markdown(f"> {paragraph}")
            else:
                st.error("Error: The specified folder path cannot be reached. Please enter a valid folder path.")


def main():
    st.title("Appropriate Job Posting Evaluation System for Resume")
    st.markdown("<h3 style='text-align: center; color: black;'>by Volkan OBAN</h3>", unsafe_allow_html=True)

    # Create a list of options in the sidebar
    page = st.sidebar.selectbox("Choose a page", ["Home", "Metrics Definitions"])
    
    # Display content based on the selected page
    if page == "Home":
        language_option = st.sidebar.selectbox("Please select a language:", ["German", "English"])
        
        # Upload the resume
        resume_option = st.selectbox("How will the resume be provided?", ["Text", "PDF"], key="resume_option_selectbox")
        resume = None
        if resume_option == "Text":
            resume = st.text_area("Please enter your resume:", key="resume_text_area")
        else:
            resume_file = st.file_uploader("Please upload a PDF file for your resume", type="pdf", key="resume_file_uploader")
            if resume_file is not None:
                resume = extract_text_from_pdf(BytesIO(resume_file.read()))

        # Create an input for the folder path containing job postings
        job_postings_folder = st.sidebar.text_input("Please provide the folder path where job postings are stored:", key="job_postings_folder_input").strip()

        if resume and job_postings_folder:
            if language_option == "German":
                run_german_version(resume, job_postings_folder)
            elif language_option == "English":
                run_english_version(resume, job_postings_folder)
        else:
            st.warning("Please provide your resume and the folder path where job postings are stored to proceed.")

    elif page == "Metrics Definitions":
        metrics_definitions()

if __name__ == "__main__":
    main()


