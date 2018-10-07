from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score 
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import os
import sys
import string
import pytesseract
from PIL import Image
from sklearn.externals import joblib
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
parser = English()

MODEL_PATH = 'model.pkl'

############################ Pipeline description start ############################

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

# Vectorization
classifier = LinearSVC()

# Using Tfidf
tfvectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer)

# Create the  pipeline to clean, tokenize, vectorize, and classify 
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', tfvectorizer),
                 ('classifier', classifier)])

############################ Pipeline description end ############################

##### Prediction #####
# Loading Model
test = joblib.load(MODEL_PATH)
# Loading a Image
text = pytesseract.image_to_string(Image.open(sys.argv[1]))
text1 = [text]

# Predict on given text
pred = test.predict(text1)
print ("It is RECEIPT" if pred == 1 else "Not a RECEIPT")


