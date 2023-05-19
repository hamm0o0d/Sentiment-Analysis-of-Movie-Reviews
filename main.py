# This is a sample Python script.

import glob
import os
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score
import pandas as pd
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punk')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


def apply_lemma(rev):
    lemma = WordNetLemmatizer()
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    lemma_sentences = []
    for sent in rev['review']:
        lemma_sent = []
        for s in sent:
            pos_text = pos_tag(s.split())
            lemma_sent.append(
                " ".join([lemma.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text]))
        lemma_sentences.append(lemma_sent)
    return lemma_sentences


def apply_porterstemmer(rev):
    stemmer = PorterStemmer()
    stemming_sentences = []
    for sent in rev['review']:
        stemmed_sent = []
        for s in sent:
            stemmed_sent.append(" ".join([stemmer.stem(word) for word in s.split()]))
        stemming_sentences.append(stemmed_sent)
    return stemming_sentences


def apply_snowballstemmer(rev):
    stemmer = SnowballStemmer(language='english')
    stemming_sentences = []
    for sent in rev['review']:
        stemmed_sent = []
        for s in sent:
            stemmed_sent.append(" ".join([stemmer.stem(word) for word in s.split()]))
        stemming_sentences.append(stemmed_sent)
    return stemming_sentences


def sentence_tokenizing(rev):
    tokenized = []
    for sent in rev['review']:
        tokenized.append(sent_tokenize(sent))
    return tokenized


def filtering_stop_words(rev):
    stop_words = set(stopwords.words('english'))
    # remove specific words from the set
    stop_words.discard('not')
    stop_words.discard('no')
    filtered_content = []
    for row in rev['review']:
        filtered_sent = []
        for sent in row:
            filtered_sentence = [word for word in sent.split() if word.casefold() not in stop_words]
            filtered_sent.append(' '.join(filtered_sentence))
        filtered_content.append(filtered_sent)

    return filtered_content


def pos_tagging(rev):
    pos_sentences = []
    for sent in rev['review']:
        tags = []
        for s in sent:
            pos_sentence = pos_tag(s.split())
            tags.append(pos_sentence)
        pos_sentences.append(tags)
    return pos_sentences


def to_lowercase(rev, target):
    rev[target] = rev[target].str.lower()
    return rev


def read_data(directory):
    x, y = [], []

    folders = ["pos", "neg"]
    for folder in folders:
        path = os.path.join(directory, folder, "*")
        sentiment = [1] if folder == "pos" else [0]

        for file in glob.glob(path, recursive=False):
            with open(file, "r") as f:
                file_content = f.read()
            x.append(file_content)
            y.append(sentiment)

    return x, y


def remove_punctuation(rev, target):
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
        rev[target] = rev[target].str.replace(contraction, expansion)
    for char in string.punctuation:
        if char != '.':
            rev[target] = rev[target].str.replace(char, '')

    return rev


def get_data(directory):
    x, y = read_data(directory)
    dataframe = pd.DataFrame({"review": x, "sentiment": y})
    dataframe = dataframe.sample(1000)
    return dataframe


data = get_data(r"C:\Users\DELL\Downloads\review_polarity\txt_sentoken")


data = to_lowercase(data, 'review')

data = remove_punctuation(data, "review")
# print(data['review'][0])

# 1. tokenize our data
data['review'] = sentence_tokenizing(data)
# print("Sentences Tokenization:")
# print(data['review'][0])

# # 2. stop word removing
data['review'] = filtering_stop_words(data)
# print(data['review'][1])


# # 3. pos tagging
# data['review'] = pos_tagging(data)
# print(data['review'][0])

# # 4. Stemming
# data['review'] = apply_Porterstemmer(data)
# # print("Stemming with Port stemmer:")
# # print(data['review'][0])
#
# data['review'] = apply_snowballstemmer(data)
# # print("Stemming with snowballstemmer:")
# # print(data['review'][0])

# # 5. lemma
data['review'] = apply_lemma(data)
# print("Stemming with lemma:")
# print(data['review'][0])

# TFIDF feature generation for a maximum of 5000 features
review = data['review'].values.tolist()
flat_list = [string for sublist in review for string in sublist]
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
X = tfidf_vect.fit_transform(flat_list)

# # Splitting data into train and validation
x_train, x_valid, y_train, y_valid = train_test_split(X, data['sentiment'], test_size=0.3, random_state=110)


# # Naive Bayes training

MultinomialNB_module = MultinomialNB(alpha=0.2)
MultinomialNB_module.fit(x_train, y_train)

x_train_prediction = MultinomialNB_module.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print('Accuracy on training data : ', training_data_accuracy)

x_test_prediction = MultinomialNB_module.predict(x_valid)
test_data_accuracy = accuracy_score(x_test_prediction, y_valid)

print('Accuracy on test data : ', test_data_accuracy)

# #SVM Classifier on Word Level TF IDF Vectors
svc_module = SVC(kernel='linear', random_state=50)
svc_module.fit(x_train, y_train)

x_train_prediction = svc_module.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print('Accuracy on training data : ', training_data_accuracy)

x_test_prediction = svc_module.predict(x_valid)
test_data_accuracy = accuracy_score(x_test_prediction, y_valid)

print('Accuracy on test data : ', test_data_accuracy)


# #LogisticRegression on Word Level TF IDF Vectors
logistic_model = LogisticRegression(C=2, random_state=11)
logistic_model.fit(x_train, y_train)

Train_prediction = logistic_model.predict(x_valid)
print('logistic Regression accuracy =', accuracy_score(Train_prediction, y_train))

ytest_prediction = logistic_model.predict(x_valid)
print('logistic Regression accuracy =', accuracy_score(ytest_prediction, y_valid))


