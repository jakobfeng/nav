#  main solution procedure
from src.Descript_cleaner import clean_ad_description
from src.make_training_set import tokenize_description
from src.preprocessor import pre_process_sentence
from src.naive_bayes import get_class_random
import pandas as pd


def preprocess(sentences):
    vectors_dict = {}
    for sentence in sentences:
        vector = pre_process_sentence(sentence)
        vectors_dict[sentence] = vector
    return vectors_dict


def classify_ad(ad):  # ad is a list of two dataframes, struct and descript
    struct_df = ad[0]
    descript_df = ad[1]
    result = pd.DataFrame(
        columns=['Stilling id', 'Registrert dato', 'Yrke grovgruppe', 'Setning', 'Pros. Setning', 'Kategori'])
    description = descript_df.iloc[0, 3]
    stilling_id_ = descript_df.iloc[0, 1]
    ad_date = struct_df.iloc[0, struct_df.columns.get_loc("Registrert dato")]
    job_group = struct_df.iloc[0, struct_df.columns.get_loc("Yrke grovgruppe")]
    clean_desc = clean_ad_description(description)
    sentences = tokenize_description(clean_desc)
    vectors = preprocess(sentences)
    for sentence, vector in vectors.items():
        category = get_class_random(vector)
        row = {"Stilling id": stilling_id_, 'Registrert dato': ad_date,
               "Yrke grovgruppe": job_group, "Setning": sentence, 'Pros. Setning': vector, "Kategori": category}
        result = result.append(row, ignore_index=True)
    return result


if __name__ == '__main__':
    path_descript = "..\\data\\training\\descript_example.csv"
    df_descript = pd.read_csv(path_descript, sep=",")
    stilling_id = df_descript.iloc[0, 1]
    path_struct = "..\\data\\training\\struct_example.csv"
    df_struct = pd.read_csv(path_struct, sep=";")
    df_descript = df_descript.tail(1)
    df_struct = df_struct.loc[df_struct["Stilling id"] == stilling_id]
    ad = [df_struct, df_descript]
    labeled_ad = classify_ad(ad)

    ad_id = 1
    labeled_ad.to_csv("..\\data\\output\\labeled_ad_" + str(ad_id) + ".csv", index=False)
