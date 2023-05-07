import glob
import os
import string
import re
import pandas as pd
from nltk import pos_tag
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


def apply_lemmatization(data):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentences = []
    for pos_sentence in data['review']:
        lemmatized_sentence = []
        for word, pos in pos_sentence:
            if pos.startswith('J'):
                # Adjective
                lemmatized_word = lemmatizer.lemmatize(word, pos='a')
            elif pos.startswith('V'):
                # Verb
                lemmatized_word = lemmatizer.lemmatize(word, pos='v')
            elif pos.startswith('N'):
                # Noun
                lemmatized_word = lemmatizer.lemmatize(word, pos='n')
            elif pos.startswith('R'):
                # Adverb
                lemmatized_word = lemmatizer.lemmatize(word, pos='r')
            else:
                # Use default lemmatization (noun)
                lemmatized_word = lemmatizer.lemmatize(word)

            lemmatized_sentence.append(lemmatized_word)
        lemmatized_sentences.append(lemmatized_sentence)
    return lemmatized_sentences


def sentence_tokenizing(data):
    tokenized = []
    for sent in data['review']:
        tokenized.append(sent_tokenize(sent))
    return tokenized


# def filtering_stop_words(data):
#     stop_words = set(stopwords.words('english'))
#     # remove specific words from the set
#     stop_words.discard('not')
#     stop_words.discard('no')
#     filtered_content = []
#     for row in data['review']:
#         filtered_sent = []
#         for sent in row:
#             filtered_sentence = [word for word in sent if word.lower() not in stop_words]
#             filtered_sent.append(filtered_sentence)
#         filtered_content.append(filtered_sent)
#
#     return filtered_content





# def filtering_stop_words(data):
#     stop_words = set(stopwords.words('english'))
#     stop_words.discard('not')
#     stop_words.discard('no')
#     filtered_content = []
#     for row in data['review']:
#         sentences = sent_tokenize(row)
#         filtered_sent = []
#         for sent in sentences:
#             words = word_tokenize(sent)
#             filtered_sentence = [word for word in words if word.lower() not in stop_words]
#             filtered_sent.append(filtered_sentence)
#         filtered_content.append(filtered_sent)
#     return filtered_content




def pos_tagging(df):
    pos_sentences = []
    for sentence in df['review']:
        pos_sentence = pos_tag(sentence)
        pos_sentences.append(pos_sentence)

    return pos_sentences


def removing_dot_after_pos_tagging(df):
    listbig = []
    for sent in df['review']:
        listy = []
        for i in sent:
            if i != '.':
                listy.append(i)

        listbig.append(listy)

    return listbig


def to_lowercase(data, target):
    data[target] = data[target].str.lower()
    return data


def read_data(dir):
    X, Y = [], []

    folders = ["pos", "neg"]
    for folder in folders:
        path = os.path.join(dir, folder, "*")
        sentiment = [1] if folder == "pos" else [0]

        for file in glob.glob(path, recursive=False):
            with open(file, "r") as f:
                file_content = f.read()
            X.append(file_content)
            Y.append(sentiment)

    return X, Y


def remove_punct(data):
    listy = []
    for sent in data['review']:
        for char in string.punctuation:
            if char != '.':
                sent = sent.replace(char, '')
        listy.append(sent)

    return listy


def get_data(dir):
    X, y = read_data(dir)
    df = pd.DataFrame({"review": X, "sentiment": y})
    return df

    # define a set of stopwords

    # def remove_punct(df, target):
    df[target] = df[target].str.replace('[^\w\s]', '')
    return df


df = get_data("C:/Users/ahmed/OneDrive/Desktop/txt_sentoken")

df = pd.DataFrame(df)
df = to_lowercase(df, 'review')

df['review'] = remove_punct(df)


# 1. tokenize our data
df['review'] = sentence_tokenizing(df)


# # 2. stop word removing
# df['review'] = filtering_stop_words(df)
# print(df['review'][50])

# print(df['review'][50])

# # 3. pos tagging
# df['review'] = pos_tagging(df)
# # 4. lemmitization
# df['review'] = apply_lemmatization(df)
# df
#


# def remove_punct(df, target):
#     df[target] = df[target].str.replace('[^\w\s]', '')
#     return df


# df_train, df_test = train_test_split(df, test_size=0.2)
# X_train, Y_train = df_train["review"], df_train["sentiment"]
# X_test, Y_test = df_test["review"], df_test["sentiment"]
