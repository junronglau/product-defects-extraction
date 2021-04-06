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
from sumy.summarizers.text_rank import TextRankSummarizer

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
    load_path = "data/uc3/defects_summarizer/generated_data_topics.csv"
    df = pd.read_csv(load_path)
    topic_df = pd.read_csv("data/uc3/defects_summarizer/generated_topic_summaries.csv")
    ticker_options = list(topic_df['category'])

    return df, topic_df, ticker_options
def app():
    st.markdown("<h1 style='text-align: center; color: teal; font-size:5vw'>Extractive summarization model</h1>", unsafe_allow_html=True)
    st.markdown('<style> .stButton>button {font-size: 20px;color: teal;}</style>', unsafe_allow_html=True)

    with st.spinner("Loading relevant data from database & models...Please wait..."):
        df, topic_df, ticker_options = load()

    st.write('')
    with st.beta_container():

        topic_input = st.selectbox('Type of defects', ticker_options)
        num_docs = st.number_input("Number of top documents to summarize", 5, 10)
        num_sentences = st.number_input("Number of sentences in summary", 1, 10)

        if st.button('Summarize reviews'):
            topic_ind = topic_df['topic'][topic_df['category'] == topic_input].item()
            df = df[df['defect_topic'] == topic_ind]
            preprocessor = TextRankPreprocessor(df, n_docs=num_docs)
            preprocessor.prepare_data()
            model = TextRankSummarizer()
            data = preprocessor.get_train_data()
            for topic, parser in data['features']:
                summary = model(parser.document, num_sentences)
                text_summary = ""
                for sentence in summary:
                    text_summary += str(sentence)
                st.write(text_summary)

            st.write(preprocessor.train_df)