from nltk.stem.snowball import NorwegianStemmer
from nltk.corpus import stopwords
import re
#  import nltk
#  nltk.download('stopwords')
from sklearn.preprocessing import LabelEncoder



stemmer = NorwegianStemmer()
stopwords = stopwords.words('norwegian', 'english')
Encoder = LabelEncoder()  # do more research?


# Helping methods used in main
def stem_word(word):
    return stemmer.stem(word)


def get_all_stop_words():
    return stopwords


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


def get_personal_words_removal_list():
    removal_list = ["du", "gode", "gjerne", "må", "lang", "innen", "innenfor", "minimum", "år", "har", "tilsvarende",
                    "bruk", "vi", "beherske", "person", "as", "våre", "kunne", "håndtere", "enhver", "tar",
                    "avdeling", "knyttet", "jobbe", "arbeid", "evne", "søk", "kontakt", "kanal", "dager", "bidra",
                    "ulike", "sette", "oppgavene", "relevant", "ønskelig", "samt", "primært", "utføre", "gjennomføre",
                    "hovedoppgave", "stor", "godt", "andre", "høy", "ta", "m", "liker", "fordel", "annen", "kjennskap",
                    "sørge", "blant", "behov", "følges", "bistå", "delta", "herunder", "eksisterende", "av", "års",
                    "samtidig", "arbeidsoppgaver", "viktigste"]
    english_rem_word = ["and", "or", "of", "in", "b", "be", "good", "with", "will", "years", "d", "have", "well", "the",
                        "to"]
    total_list = removal_list + english_rem_word
    return total_list


def get_geo_words():
    geo_words = ['Finnmark', 'Alta', 'Hammerfest', 'Honningsvåg', 'Kirkenes', 'Vadsø', 'Vardø', 'Troms', 'Finnsnes', 'Harstad', 'Tromsø', 'Nordland', 'Bodø', 'Brønnøysund', 'Fauske', 'Leknes', 'Mo i Rana', 'Mosjøen', 'Narvik', 'Sandnessjøen', 'Sortland', 'Stokmarknes', 'Svolvær', 'Nord-Trøndelag', 'Levanger', 'Namsos', 'Nærøy', 'Steinkjer', 'Stjørdal', 'Verdalsøra', 'Sør-Trøndelag', 'Trondheim', 'Brekstad', 'Orkanger ', 'Møre og Romsdal', 'Fosnavåg', 'Kristiansund', 'Molde', 'Ulsteinvik', 'Ålesund', 'Åndalsnes', 'Sogn og Fjordane', 'Flora', 'Førde', 'Måløy', 'Hordaland', 'Bergen', 'Stord', 'Odda', 'Rogaland', 'Bryne', 'Eigersund', 'Haugesund', 'Jørpeland', 'Kopervik', 'Sandnes', 'Sauda', 'Skudeneshavn', 'Stavanger', 'Åkrehamn', 'Vest-Agder', 'Flekkefjord', 'Kristiansand', 'Farsund', 'Lyngdal', 'Mandal', 'Aust-Agder', 'Arendal', 'Grimstad', 'Lillesand', 'Risør', 'Tvedestrand', 'Telemark', 'Brevik', 'Kragerø', 'Langesund', 'Notodden', 'Porsgrunn', 'Rjukan', 'Skien', 'Stathelle', 'Vestfold', 'Holmestrand', 'Horten', 'Larvik', 'Sandefjord', 'Stavern', 'Svelvik', 'Åsgårdstrand', 'Tønsberg', 'Buskerud', 'Drammen', 'Hokksund', 'Hønefoss', 'Ringerike', 'Kongsberg', 'Oppland', 'Fagernes', 'Gjøvik', 'Lillehammer', 'Otta', 'Vinstra', 'Raufoss', 'Hedmark', 'Elverum', 'Hamar', 'Kongsvinger', 'Brumunddal', 'Moelv', 'Oslo', 'Akershus', 'Drøbak', 'Lillestrøm', 'Sandvika', 'Ski', 'Jessheim ', 'Østfold', 'Askim', 'Fredrikstad', 'Halden', 'Moss', 'Mysen', 'Sarpsborg']
    geo_words = [city.lower() for city in geo_words]
    return geo_words


if __name__ == '__main__':
    geo_words_ = get_geo_words()

    print(geo_words_)
