from nltk.stem.snowball import NorwegianStemmer
from nltk.corpus import stopwords
#  import nltk
#  nltk.download('stopwords')
from sklearn.preprocessing import LabelEncoder

stemmer = NorwegianStemmer()
stopwords = stopwords.words('norwegian')
Encoder = LabelEncoder()  # do more research?


def pre_tokenize(sentence):
    return sentence.split()


def pre_stem(sentence):
    return [stemmer.stem(word) for word in sentence]


def pre_word_removal(sentence):
    return [word for word in sentence if word not in stopwords]


def pre_vectorize(sentence):  # not complete. Is it necessary?
    return sentence


def pre_process_sentence(sentence):
    s = pre_tokenize(sentence)
    s = pre_stem(s)
    s = pre_word_removal(s)
    v = pre_vectorize(s)
    return v
