
import re 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def preprocess_text(text):
  text = text.lower()
  ### REMOVE ANY SPECIAL CHARACHTERS
  text=re.sub(r'[^a-zA-Z\s]', '', text)
  tokens =  word_tokenize(text)
  stop_words = set(stopwords.words('english'))
  filtered_tokens = [word for word in tokens if word not in stop_words]
  stemmer = PorterStemmer()
  stemmed_tokens =[stemmer.stem(word) for word in filtered_tokens]
  stemmed_tokens=' '.join(stemmed_tokens)
  return stemmed_tokens