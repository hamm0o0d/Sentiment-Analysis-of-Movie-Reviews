import string
from nltk import pos_tag
import streamlit as sl
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

tfidf=pd.read_pickle('tfidf.pkl')
svm=pd.read_pickle('svm.pkl')
logistic=pd.read_pickle('logistic.pkl')
naive=pd.read_pickle('naive_bayes.pkl')

def remove_punctuation(rev):
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "we'd": "we would",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where's": "where is",
        "who'll": "who will",
        "who's": "who is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }
    for contraction, expansion in contractions.items():
        rev = rev.replace(contraction, expansion)
    for char in string.punctuation:
        if char != '.':
            rev = rev.replace(char, '')

    return rev

def to_lowercase(rev):
    rev= rev.lower()
    return rev

def filtering_stop_words(rev):
    stop_words = set(stopwords.words('english'))
    # remove specific words from the set
    stop_words.discard('not')
    stop_words.discard('no')
    tokens = word_tokenize(rev)
    filtered_review = [word for word in tokens if word.casefold() not in stop_words]
    filtered_review=' '.join(filtered_review)
    return filtered_review

def apply_lemma(review):
    lemma = WordNetLemmatizer()
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    tokens = word_tokenize(review)
    pos_text = pos_tag(tokens)
    lemma_review = " ".join([lemma.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text])
    return lemma_review

sl.title('Welcome to Movie Review Sentiment Analysis')
with sl.form('Write the review'):
    review = sl.text_area('User Review')

    stat = sl.form_submit_button('view review statment')
    if stat == 1:

        review = to_lowercase(review)
        review = remove_punctuation(review)

        review = filtering_stop_words(review)

        review = apply_lemma(review)

        x = tfidf.transform([review])
        y = logistic.predict(x)

        if int(y) == 1:
            sl.write("Review Statement : **Positive review** ")
        else:
            sl.write("Review Statement : **Negative review** ")




