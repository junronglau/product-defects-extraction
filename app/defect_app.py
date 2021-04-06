"""
Credits to [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps)
Lets user input a review/comment in a form of text
App will then determine if review is
1) Contains defects
2) If yes, what kind of defect category
3) The summary of defects in that category

Also allows user to generate a few random reviews from defect category and then summarize it
"""
from pathlib import Path
import sys

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from dataloader.ds_data_loader import DataLoader
from preprocess.corex_preprocessor import CorexPreprocessor
from preprocess.textrank_preprocessor import TextRankPreprocessor
from models.corex_model import CorexModel
from corextopic import corextopic as ct

from models.textrank_model import TextRankModel
from trainers.corex_trainer import CorexTrainer
from trainers.textrank_trainer import TextRankTrainer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.clean import clean_text
from utils.utils import get_args, app_preprocess
from utils.config import process_config


@st.cache(allow_output_mutation=True)
def load():
    load_path = "data/uc3/defects_summarizer/topic_vectorizer.pkl"
    topic_vectorizer = pickle.load(open(load_path, 'rb'))

    load_path = "data/uc3/defects_summarizer/summary_model.pkl"
    topic_model = pickle.load(open(load_path, 'rb'))
    topic_df = pd.read_csv("data/uc3/defects_summarizer/generated_topic_summaries.csv")

    load_path = "data/uc3/defects_classifier/defect_classifier.pkl"
    defect_model = pickle.load(open(load_path, 'rb'))

    load_path = "data/uc3/defects_summarizer/defect_vectorizer.pkl"
    defect_vectorizer = pickle.load(open(load_path, 'rb'))

    return topic_vectorizer, topic_model, topic_df, defect_model, defect_vectorizer

def app():
    st.markdown("<h1 style='text-align: center; color: teal; font-size:5vw'>Defect extraction model</h1>", unsafe_allow_html=True)
    st.markdown('<style> .stButton>button {font-size: 20px;color: teal;}</style>', unsafe_allow_html=True)

    with st.spinner("Loading relevant data from database & models...Please wait..."):
        topic_vectorizer, topic_model, topic_df, defect_model, defect_vectorizer = load()

    st.write('')
    with st.beta_container():

        text_input = st.text_input(label="Input your review")
        num_topics = st.number_input("Number of candidate topics",1, 10)

        if st.button('Extract defects'):
            if not text_input.strip():
                st.error("No text was entered")
            else:
                df = pd.DataFrame(data=[text_input], columns=['comment'])
                df['cleaned_text'] = clean_text(df['comment'], "data/uc3/labels_generator/contractions.txt", "data/uc3/labels_generator/slangs.txt")

                # First check if defects or not
                # features = defect_vectorizer.transform(text_input)
                features = pd.DataFrame(defect_vectorizer.transform(df['cleaned_text']).toarray())
                predicted = defect_model.predict(features)[0]

                if predicted == 1:  # If defect is predicted
                    st.header(f"Top {num_topics} topics (in order)")
                    data = app_preprocess(df)
                    data = topic_vectorizer.transform(data['cleaned_text'])
                    mask = np.array(topic_model.predict_proba(data))[0][0]
                    ind = (-mask).argsort()[:num_topics]
                    st.write(topic_df.loc[topic_df.index[list(ind)]])
                else:
                    st.header(f"There are no defects identified in the review")