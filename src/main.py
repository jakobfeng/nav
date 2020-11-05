# Script for running model for the perspective of a user
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from src.Ad_classifier import classify_ad
from src.naive_bayes import load_model
from src.preprocessor import stem_word
from src.preprocessor import get_all_stop_words
from src.preprocessor import pre_tokenize
from src.preprocessor import get_personal_words_removal_list
from src.preprocessor import get_geo_words
from src.word_cloud_visualizer import plot_frq_dict

stopwords = get_all_stop_words()  # list used to remove stopwords when analysing results
removal_words = get_personal_words_removal_list()
geo_words = get_geo_words()


def run(industry_list, regions, start, end, classes):
    verify_valid_dates(start, end)
    path = create_path_for_results(industry_list, regions, start, end)
    results_already_exists = check_if_result_already_exists(path)
    if results_already_exists:
        result_df = pd.read_csv(path, sep=",")
    else:
        region_list = get_region_list(regions)
        df = get_df_within_interval(start, end)
        df = get_df_filtered_on_industry_and_region(industry_list, region_list, df)
        print("Number of ads to classify: {}\n".format(len(df)))
        result_df = pd.DataFrame(
            columns=['Stilling id', 'Registrert dato', 'Yrke grovgruppe', 'Setning', 'Pros. Setning', 'Kategori'])
        model = load_model(1)
        for index, ad in df.iterrows():
            labeled_ad = classify_ad(model, ad)
            result_df = result_df.append(labeled_ad, ignore_index=True)
        result_df = result_df.sort_values(by=['Kategori'])
        save_df_results_to_file(result_df, industry_list, regions, start, end)

    list_of_df_classes = []
    for c in classes:
        class_df = result_df[result_df["Kategori"] == c]
        list_of_df_classes.append(class_df)

    analyze_results_from_classes_dfs(list_of_df_classes, path)


# Helping method for checking if result path already exists
def check_if_result_already_exists(path):
    directory = "..\\data\\output"
    existing_paths = sorted(Path(directory).iterdir())
    return path in [str(p) for p in existing_paths]


# Method for analyzing results from list of dataframes for each class
def analyze_results_from_classes_dfs(list_of_df_classes, path):
    list_of_dictionaries_classes = []
    for class_df in list_of_df_classes:
        word_count_dict = get_word_count_dictionary(class_df)
        word_count_dict = remove_word_from_dictionary(word_count_dict)
        list_of_dictionaries_classes.append(word_count_dict)
    for i in range(len(list_of_dictionaries_classes)):
        plot_frq_dict(list_of_dictionaries_classes[i], i+1, path)


# Helping method for removing words from frequency dictionary
def remove_word_from_dictionary(word_count_dict):
    words_to_remove = []
    for word in word_count_dict.keys():
        stemmed_word = stem_word(word)
        for rem_word in removal_words:
            if stemmed_word == stem_word(rem_word):
                words_to_remove.append(word)
        if word in geo_words:
            words_to_remove.append(word)
    word_count_dict = {k: v for k, v in word_count_dict.items() if k not in words_to_remove}
    return word_count_dict



# Helping method for analyzing overall results from class_df
def get_word_count_dictionary(class_df):
    freq_count = {}
    for index, row in class_df.iterrows():
        sentence = row[3]
        tok_sentence = pre_tokenize(sentence)
        for word in tok_sentence:
            word = word.lower() # lower case
            if word not in stopwords:
                if word in freq_count.keys():
                    freq_count[word] += 1
                else:
                    stemmed_word = stem_word(word)
                    found_match = False
                    for key in freq_count.keys():
                        stemmed_key = stem_word(key)
                        if stemmed_key == stemmed_word:
                            freq_count[key] += 1
                            found_match = True
                    if not found_match:
                        freq_count[word] = 1
    freq_count = {k: v for k, v in freq_count.items() if v != 1}
    return freq_count


# Helping method for creating path
def create_path_for_results(industry_list, region_list, start, end):
    industry = "_".join(industry_list)
    region = "_".join(region_list)
    path = "..\\data\\output\\{}_{}_{}_{}.csv".format(industry, region, start, end)
    return path


# Helping method to save result df to file, sorted on category
def save_df_results_to_file(df, industry_list, region_list, start, end):
    industry = "_".join(industry_list)
    region = "_".join(region_list)
    path = "..\\data\\output\\{}_{}_{}_{}.csv".format(industry, region, start, end)
    df.to_csv(path, sep=",", index=False)


# Helping method for getting dataframe filtered on industry and region
def get_df_filtered_on_industry_and_region(industry_list, region_list, df):
    df = df[df["Arbeidssted fylke"].isin(region_list)]
    df = df[df["Yrke grovgruppe"].isin(industry_list)]
    return df


# Helping method for getting dataframe within time interval
def get_df_within_interval(start, end):
    df_years_struct = []
    df_years_descript = []
    for year in range(start.year, end.year + 1):
        path_struct = "..\\data\\input\\struct\\{}_data.csv".format(year)
        df_struct = pd.read_csv(path_struct, sep=";", usecols=["Stilling id", "Registrert dato", "Arbeidssted fylke",
                                                               "Arbeidssted land", "Yrke grovgruppe"])
        df_years_struct.append(df_struct)
        path_descript = "..\\data\\input\\descript_cl\\{}_descript_cl.csv".format(year)
        df_descript = pd.read_csv(path_descript, sep=",", usecols=["Stilling Id", "Stillingsbeskrivelse vasket"])
        df_descript = df_descript.rename(columns={"Stilling Id": "Stilling id"})
        df_years_descript.append(df_descript)

    df_struct = df_years_struct[0]
    df_descript = df_years_descript[0]
    for i in range(1, len(df_years_struct)):
        df_struct = df_struct.append(df_years_struct[i], ignore_index=True)
        df_descript = df_descript.append(df_years_descript[i], ignore_index=True)
    df = pd.merge(df_descript, df_struct, on="Stilling id", how="outer")
    df = df[df["Stillingsbeskrivelse vasket"].notna()]
    date_format = "%Y-%m-%d"
    df['Registrert dato'] = pd.to_datetime(df['Registrert dato'], format=date_format)
    df['Registrert dato'] = df['Registrert dato'].dt.date
    df = df[(df['Registrert dato'] >= np.datetime64(start)) & (df['Registrert dato'] <= np.datetime64(end))]
    return df


# Helping method checking valid dates
def verify_valid_dates(start, end):
    if start > end:
        msg = "Msg: Start date must be before end date"
        print('\033[91m' + msg + '\033[0m')
        assert False
    elif start.year < 2013:
        msg = "Msg: Start date must be 2013 or after"
        print('\033[91m' + msg + '\033[0m')
        assert False
    elif end.year > 2018:
        msg = "Msg: End date must be 2018 or before"
        print('\033[91m' + msg + '\033[0m')
        assert False


# Helping method for returning list of current + old region names
def get_region_list(region):
    updated_regions = []
    region_update_dict = {"Agder": ["Vest-Agder", "Aust-Agder"], "Vestland": ["Hordaland", "Sogn og Fjordane"],
                          "Vestfold og Telemark": ["Vestfold", "Telemark"], "Innlandet": ["Hedmark", "Oppland"],
                          "Viken": ["Østfold", "Akershus", "Buskerud"],
                          "Trøndelag": ["Sør-Trøndelag", "Nord-Trøndelag"],
                          "Troms og Finnmark": ["Troms", "Finnmark"]}
    regions_needing_update = list(region_update_dict.keys())
    for r in region:
        updated_regions.append(r)
        if r in regions_needing_update:
            updated_regions.extend(region_update_dict[r])
    return updated_regions


if __name__ == '__main__':
    industries = {0: "Ingen yrkesbakgrunn eller uoppgitt", 1: "Ledere", 2: "Ingeniør- og ikt-fag", 3: "Undervisning",
                  4: "Akademiske yrker", 5: "Helse, pleie og omsorg", 6: "Barne- og ungdomsarbeid",
                  7: "Meglere og konsulenter",
                  8: "Kontorarbeid", 9: "Butikk- og salgsarbeid", 10: "Jordbruk, skogbruk og fiske",
                  11: "Bygg og anlegg",
                  12: "Industriarbeid", 13: "Reiseliv og transport", 14: "Serviceyrker og annet arbeid"}

    current_regions = {0: "Troms og Finnmark", 1: "Nordland", 2: "Trøndelag", 3: "Møre og Romsdal", 4: "Vestland",
                       5: "Rogaland", 6: "Agder", 7: "Vestfold og Telemark", 8: "Viken", 9: "Oslo", 10: "Innlandet"}

    # -----------------------------------------------------------------------
    industry_ = [8]
    region_ = [2]
    start_date_ = datetime.date(2017, 11, 1)
    end_date_ = datetime.date(2017, 11, 30)
    classes_ = [1, 2, 3]
    # _______________________________________________________________________
    industry_input = [industries[i] for i in industry_]
    region_input = [current_regions[i] for i in region_]

    run(industry_input, region_input, start_date_, end_date_, classes_)
