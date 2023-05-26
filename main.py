# This is a sample Python script.
import pickle
import glob
import os
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, classification_report


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Lemmatization function

def apply_lemma(rev):
    lemma = WordNetLemmatizer()
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    lemma_reviews = []
    for row in rev['review']:
        tokens = word_tokenize(row)
        pos_text = pos_tag(tokens)
        lemma_reviews.append(" ".join([lemma.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text]))
    return lemma_reviews


# Port-stemming function

def apply_porterstemmer(rev):
    stemmer = PorterStemmer()
    stemming_reviews = []
    for row in rev['review']:
        tokens = word_tokenize(row)
        stemming_reviews.append(" ".join([stemmer.stem(word) for word in tokens]))
    return stemming_reviews

# Snowball-Stemming function

def apply_snowballstemmer(rev):
    stemmer = SnowballStemmer(language='english')
    stemming_reviews = []
    for row in rev['review']:
        tokens = word_tokenize(row)
        stemming_reviews.append(" ".join([stemmer.stem(word) for word in tokens]))
    return stemming_reviews

# Removing all the stop_words

def filtering_stop_words(rev):
    stop_words = set(stopwords.words('english'))
    # remove specific words from the set
    stop_words.discard('not')
    stop_words.discard('no')
    filtered_reviews = []
    for row in rev['review']:
        tokens = word_tokenize(row)
        filtered_sentence = [word for word in tokens if word.casefold() not in stop_words]
        filtered_reviews.append(' '.join(filtered_sentence))
    return filtered_reviews

# find the pos-tag for each word in reviews

def pos_tagging(rev):
    pos_reviews = []
    for row in rev['review']:
        tokens = word_tokenize(row)
        pos_rev = pos_tag(tokens)
        pos_reviews.append(pos_rev)
    return pos_reviews

# convert any apper-case to lower-case in all reviews

def to_lowercase(rev, target):
    rev[target] = rev[target].str.lower()
    return rev

# read the reviews ,and it's sentiment from files in the poss and neg folders

def read_data(directory):
    x, y = [], []

    folders = ["pos", "neg"]
    for folder in folders:
        path = os.path.join(directory, folder, "*")
        sentiment = 1 if folder == "pos" else 0

        for file in glob.glob(path, recursive=False):
            with open(file, "r") as f:
                file_content = f.read().strip()
            x.append(file_content)
            y.append(sentiment)

    return x, y

# Cleaning and remove all the punctuation from the reviews

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

# receive the reviews and it's sentiment as a dataframe

def get_data(directory):
    x, y = read_data(directory)
    dataframe = pd.DataFrame({"review": x, "sentiment": y})
    return dataframe

data = get_data(r"C:\Users\ahmed\OneDrive\Desktop\txt_sentoken")

# Apply  Preprocessing
data = to_lowercase(data, 'review')

data = remove_punctuation(data, "review")
# print(data['review'][0])

# # 1. stop word removing
data['review'] = filtering_stop_words(data)
# print(data['review'][0])


# # 2. pos tagging
# data['review'] = pos_tagging(data)
# print(data['review'][0])

# # 3. Stemming
# data['review'] = apply_porterstemmer(data)
# print(data['review'][0])

# data['review'] = apply_snowballstemmer(data)
# print(data['review'][0])

# # 4. lemma and pos-tagging
data['review'] = apply_lemma(data)
# print(data['review'][0])

# # TFIDF feature generation for a maximum of 1000 features
reviews = data["review"]
sentiments = data["sentiment"]
tfidf = TfidfVectorizer()#max_features=1000
X = tfidf.fit_transform(reviews)

# saving tfidf
with open('tfidf.pkl','wb') as f:
    pickle.dump(tfidf,f)

# # # Splitting data into train and validation data
x_train, x_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=110)

# Apply three of classification models (Naive Bayes,SVC,logistic regression)
# # # Naive Bayes Classifier on Word Level TF IDF Vectors
MultinomialNB_module = MultinomialNB(alpha=0.2)
MultinomialNB_module.fit(x_train, y_train)


Naive_Train_prediction = MultinomialNB_module.predict(x_train)
print('Naive Bayes accuracy of training data : ', accuracy_score(Naive_Train_prediction, y_train))

Naive_Test_prediction = MultinomialNB_module.predict(x_test)
print('Naive Bayes accuracy of testing data : ', accuracy_score(Naive_Test_prediction, y_test))
# print(classification_report(y_test, Test_prediction))


# # #SVM Classifier on Word Level TF IDF Vectors
svc_module = SVC(kernel='linear')
svc_module.fit(x_train, y_train)

SVC_Train_prediction = svc_module.predict(x_train)
print('SVC accuracy of training data : ', accuracy_score(SVC_Train_prediction, y_train))

SVC_Test_prediction = svc_module.predict(x_test)
print('SVC accuracy of testing data  : ', accuracy_score(y_test, SVC_Test_prediction))
# print(classification_report(y_test, Test_prediction))


# # #LogisticRegression Model on Word Level TF IDF Vectors
logistic_model = LogisticRegression(C=2)
logistic_model.fit(x_train, y_train)

Logistic_Train_prediction = logistic_model.predict(x_train)
print('logistic Regression accuracy of training data=', accuracy_score(Logistic_Train_prediction, y_train))

Logistic_Test_prediction = logistic_model.predict(x_test)
print('logistic Regression accuracy of testing data=', accuracy_score(Logistic_Test_prediction, y_test))
# print(classification_report(y_test, Test_prediction))

# saving models:
with open('logistic.pkl','wb') as f:
    pickle.dump(logistic_model,f)

with open('naive_bayes.pkl', 'wb') as f:
        pickle.dump(MultinomialNB_module, f)

with open('svm.pkl', 'wb') as f:
            pickle.dump(svc_module, f)


# Result Visualization
# Plot bar chart
# Define the names of the models and their corresponding accuracy scores
naive = accuracy_score(Naive_Test_prediction, y_test)
svc = accuracy_score(y_test, SVC_Test_prediction)
logistic = accuracy_score(Logistic_Test_prediction, y_test)
model_names = ["Naive Bayes", "SVM", "Logistic Regression"]
accuracy_scores = [naive, svc, logistic]

# Create a bar chart for the accuracy scores
plt.bar(model_names, accuracy_scores)
plt.title("Classification Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.0, 1.0)
plt.show()
# Plot confusion matrix for each model
models = [("Naive Bayes", MultinomialNB_module),
          ("SVC", svc_module),
          ("Logistic Regression", logistic_model)]

for name, model in models:
    predictions = model.predict(x_test)
    cm = confusion_matrix(y_test, predictions)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(["Negative", "Positive"]))
    plt.xticks(tick_marks, ["Negative", "Positive"])
    plt.yticks(tick_marks, ["Negative", "Positive"])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Plot ROC curve for each model
models = [("Naive Bayes", MultinomialNB_module),
          ("SVC", svc_module),
          ("Logistic Regression", logistic_model)]

for name, model in models:
    if name == "Naive Bayes":
        y_scores = model.predict_proba(x_test)[:, 1] # Use probability of positive class as score
    else:
        y_scores = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc_score = roc_auc_score(y_test, y_scores)
    plt.plot(fpr, tpr, label=name + f" (AUC={auc_score:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Generate classification report for each model
for name, model in models:
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(f"{name} Classification Report:\n{report_df}\n")


# This is an example of negative and positive reviews and the model try to predict if it is a positive or negative
x_example = 'this movie is really amazing and wonderful movie that i have ever seen in my life '
y_example = 'i hate this movie so much , it is a boring movie and so disgusting and i did not complete it'

vec = tfidf.transform([y_example])
result = svc_module.predict(vec)
if result == 1:
    print("the sentiment prediction of this review is: Poss")
else:
    print("the sentiment prediction of this review is: Neg")

