import glob
import os
import string
import re
import pandas as pd
from nltk import pos_tag
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')








def apply_lemmatization(data):
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    lemmatized_sentences = []
    for sent in data['review']:
        lemmatized_sent = []
        for subsent in sent:
            pos_text = pos_tag(subsent.split())
            lemmatized_sent.append(" ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text]))
        lemmatized_sentences.append(lemmatized_sent)
    return lemmatized_sentences



def apply_stemming(data):
    stemmer = PorterStemmer()
    stemming_sentences = []
    for sent in data['review']:
        stemmed_sent = []
        for subsent in sent:
            stemmed_sent.append(" ".join([stemmer.stem(word) for word in subsent.split()]))
        stemming_sentences.append(stemmed_sent)
    return stemming_sentences


def sentence_tokenizing(data):
    tokenized = []
    for sent in data['review']:
        tokenized.append(sent_tokenize(sent))
    return tokenized




def filtering_stop_words(data):
    stop_words = set(stopwords.words('english'))
    # remove specific words from the set
    stop_words.discard('not')
    stop_words.discard('no')
    filtered_content = []
    for row in data['review']:
        filtered_sent = []
        for sent in row:
            filtered_sentence = [word for word in sent.split() if word.casefold() not in stop_words]
            filtered_sent.append(' '.join(filtered_sentence))
        filtered_content.append(filtered_sent)

    return filtered_content



def pos_tagging(df):
    pos_sentences=[]
    for sent in df['review']:
        tages = []
        for subsent in sent:
            pos_sentence = pos_tag(subsent.split())
            tages.append(pos_sentence)
        pos_sentences.append(tages)
    return pos_sentences





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




df = get_data(r"C:\Users\DELL\Downloads\review_polarity\txt_sentoken")

df = pd.DataFrame(df)
df = to_lowercase(df, 'review')


df['review'] = remove_punct(df)
# print(df['review'][1])

# 1. tokenize our data
df['review'] = sentence_tokenizing(df)
# print(df['review'][1])

# # 2. stop word removing
df['review'] = filtering_stop_words(df)
# print(df['review'][1])


# # 3. pos tagging
# df['review'] = pos_tagging(df)
# print(df['review'][0])

# # 4. Stemming
# df['review'] = apply_stemming(df)
# print(df['review'][0])

# # 5. lemmitization
df['review'] = apply_lemmatization(df)
# print(df['review'][1])






# df_train, df_test = train_test_split(df, test_size=0.2)
# X_train, Y_train = df_train["review"], df_train["sentiment"]
# X_test, Y_test = df_test["review"], df_test["sentiment"]


## removing all numbers in the dataset
## asking the TA about the stemming
## asking the TA about running the code