from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score 
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import os
import string
from sklearn.externals import joblib
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
parser = English()

# Path to the input data which inclues textfiles(images to OCR)
INPUT_PATH = './TEXT_DATA/'

#Custom transformer using spaCy 
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

# Build a list of stopwords to use to filter
stopwords = list(STOP_WORDS)
punctuations = string.punctuation

# Basic function to clean the text 
def clean_text(text):     
    return text.strip().lower()

# Custom tokenizer using spaCy 
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    return mytokens

#  Create labels to the word file lable for "RECEIPT" is 1 and for "NOT RECEIPT" it is 0 
def create_features_labels(INPUT_PATH):
    direc = INPUT_PATH
    files = os.listdir(direc)
    text_files = [direc + text for text in files]

    feature_set = []
    labels = []

    for text in text_files[1:]:
        #print(text)
        data = []
        f = open(text)
        words = f.read().lower()
        #print(words)
        feature_set.append(words)

        if (text.split('/')[-1]).startswith("POS"):
            labels.append(1)
        else:
            labels.append(0)
    return (feature_set, labels)

# Save the model
def save_model(clf, name):
    with open(name, 'wb') as fp:
        c.dump(clf, fp)
    print ("saved")


# Classifier
classifier = LinearSVC()

# Using Tfidf Vectorization
tfvectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer)
# Features and Labels
X, ylabels = create_features_labels(INPUT_PATH)

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)

# Create the  pipeline to clean, tokenize, vectorize, and classify the text
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', tfvectorizer),
                 ('classifier', classifier)])

# Fit our data
pipe.fit(X_train,y_train)
# Test Accuracy
print("Test Accuracy: ",pipe.score(X_test,y_test))
# Dumping model into file
joblib.dump(pipe, 'model.pkl')



