import numpy as np

import nltk
from nltk import word_tokenize, pos_tag, WordNetLemmatizer, RegexpParser, TweetTokenizer
from nltk.corpus import stopwords, wordnet

import spacy

import string
import re

# downoad the tokenizer, POS-tagger, universal tag set (for more standard conventions), and the lemmatizer library
nltk.download("punkt")
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download("wordnet")
nltk.download('stopwords')

from text_processing.text_processor import TextProcessor
from text_processing.tweet_processor import TweetProcessor

import text_processing.spacy_doc_methods