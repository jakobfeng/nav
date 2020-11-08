# Script for running model for the perspective of a user
import datetime
import os
import pandas as pd
import numpy as np
from pathlib import Path
from calendar import monthrange
from dateutil.relativedelta import relativedelta
from ast import literal_eval
from src.Ad_classifier import classify_ad
from src.naive_bayes import load_model
from src.preprocessor import stem_word
from src.preprocessor import get_all_stop_words
from src.preprocessor import pre_tokenize
from src.preprocessor import get_personal_words_removal_list
from src.preprocessor import get_geo_words
from src.plot_results import plot_frq_dict
from src.plot_results import plot_frequency_df
from src.plot_results import plot_trends_df


stopwords = get_all_stop_words()  # list used to remove stopwords when analysing results
removal_words = get_personal_words_removal_list()
geo_words = get_geo_words()


# Method for creating results -------------------------------------
def create_results(industry_list, regions, start, end, classes, path):
    verify_valid_dates(start, end)
    results_already_exists = check_if_result_already_exists(path)
    if results_already_exists:
        result_df = pd.read_csv(path, sep=",")
    else:
        region_list = get_region_list(regions)
        df = get_df_within_interval(start, end)
        df = get_df_filtered_on_industry_and_region(industry_list, region_list, df)
        df.to_csv("..\\data\\output\\ads_2018_butikk_nordland.csv")
        print("Number of ads to classify: {}\n".format(len(df)))
        result_df = pd.DataFrame(
            columns=['Stilling id', 'Registrert dato', 'Yrke grovgruppe', 'Setning', 'Pros. Setning', 'Kategori'])
        model = load_model(1)
        for index, ad in df.iterrows():
            print("Classify ad no. :" + str(index+1))
            labeled_ad = classify_ad(model, ad)
            result_df = result_df.append(labeled_ad, ignore_index=True)
        result_df = result_df.sort_values(by=['Kategori'])
        save_df_results_to_file(result_df, industry_list, regions, start, end)
    result_df = result_df[result_df["Kategori"].isin(classes)]
    result_df = result_df.sort_values(by=['Kategori'])
    date_format = "%Y-%m-%d"
    result_df['Registrert dato'] = pd.to_datetime(result_df['Registrert dato'], format=date_format)
    return result_df


# Helping method for checking if result path already exists
def check_if_result_already_exists(path):
    directory = "..\\data\\output"
    existing_paths = sorted(Path(directory).iterdir())
    return path in [str(p) for p in existing_paths]


# Helping method for creating result path
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
    if len(region_list) > 0:
        df = df[df["Arbeidssted fylke"].isin(region_list)]
    if len(industry_list) > 0:
        df = df[df["Yrke grovgruppe"].isin(industry_list)]
    df = df.reset_index()
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
    print("Length of descript: {}, length of struct: {}".format(len(df_descript), len(df_struct)))
    df = pd.merge(df_descript, df_struct, on="Stilling id", how="inner")
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


# Helping method for returning start and end days
def get_start_and_end_date(year, month, months_lookback):
    final_day_of_month = monthrange(year, month)[1]
    end = datetime.date(year, month, final_day_of_month)
    start = end - relativedelta(months=months_lookback)
    start = start.replace(day=1)
    return start, end


# # Method for plotting word clouds ----------------------------------------------------
def plot_word_cloud_from_dict(df, classes, industry_list, regions, start, end, path):
    max_ads = 2000
    if len(df) > max_ads:
        df = df.sample(n=max_ads, random_state=1)
    df = df.sort_values(by=["Kategori"])
    df = df.reset_index()
    print("Making word cloud from {} number of ads\n".format(len(df)))
    list_of_df_classes = []
    for c in classes:
        class_df = df[df["Kategori"] == c]
        list_of_df_classes.append(class_df)
    list_of_dictionaries_classes = []
    for class_df in list_of_df_classes:
        word_count_dict = get_word_count_dictionary(class_df)
        word_count_dict = remove_word_from_dictionary(word_count_dict)
        list_of_dictionaries_classes.append(word_count_dict)
    directory = get_dir_for_plots(industry_list, regions, start, end)
    for i in range(len(list_of_dictionaries_classes)):
        plot_frq_dict(list_of_dictionaries_classes[i], i+1, industry_list, regions, start, end, directory)


# Helping method for creating directory for saving plots
def get_dir_for_plots(industry_list, regions, start, end):
    industry = "_".join(industry_list)
    region = "_".join(regions)
    path_dir = "..\\plots\\{}_{}_{}_{}".format(industry, region, start, end)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    return path_dir + "\\"


# Helping method for analyzing overall results from class_df
def get_word_count_dictionary(class_df):
    freq_count = {}
    for index, row in class_df.iterrows():
        print("Word freq. ad no. " + str(index+1))
        sentence = row["Setning"]
        tok_sentence = pre_tokenize(sentence)
        for word in tok_sentence:
            word = word.lower()  # lower case
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


# Method for plotting specified terms frequency -------------------------------------
def plot_frequency_word_list(results_df, industries, regions, start, end, classes, word_list, path):
    results_df = results_df.reset_index()
    print("Plotting word frequency from {} number of ads\n".format(len(results_df)))
    frequency_df = get_frequency_df(results_df, start, end, word_list)
    directory = get_dir_for_plots(industries, regions, start, end)
    plot_frequency_df(frequency_df, industries, classes, word_list, regions, directory)


def get_frequency_df(df, start, end, word_list):
    frequency_df = pd.DataFrame(columns=["Week"].extend(word_list))
    date_list = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days+1)]
    number_of_weeks = len(date_list)//7
    for week in range(number_of_weeks):
        print("Week no. " + str(week))
        idx = range(week*7, week*7+7)
        week_dates = [date_list[i] for i in idx]
        row = {"Week": week, "Date": week_dates[0]}
        for word in word_list:
            frequency = get_word_frequency(word, week_dates, df)
            row[word] = frequency
        frequency_df = frequency_df.append(row, ignore_index=True)
    return frequency_df


def get_word_frequency(word, week_dates, df):
    first = np.datetime64(week_dates[0])
    last = np.datetime64(week_dates[-1])
    subset_df = df[(df['Registrert dato'] >= first) & (df['Registrert dato'] <= last)]
    freq = 0
    stemmed_word = stem_word(word).lower()
    for index, row in subset_df.iterrows():
        pros_line = row["Pros. Setning"]
        if stemmed_word in pros_line:
            freq += 1
    return freq


# Method for detecting trends -------------------------------------
def find_trends_from_df(df, industries, regions, start, end, classes, path):
    df = df.reset_index()
    print("Finding trends from {} number of ads\n".format(len(df)))
    df_week = df.groupby(pd.Grouper(freq='W'))
    print(df_week)
    word_list = []
    column_series = literal_eval(df_week[df_week["Setning"] == 0]["Vector"]).tolist()
    for sentence in column_series:
        for word in sentence.split():
            if stem_word(word) not in [stem_word(s) for s in word_list]:
                word_list.append(word)
    print(column_series)
    frequency_df_weekly = get_frequency_df(df, start, end, word_list)
    trend_df = make_trends_df_based_on_frequency(frequency_df_weekly)
    trending_words = trend_df.columns.tolist()[2:]
    plot_trends_df(trend_df, industries, classes, trending_words, regions)


def make_trends_df_based_on_frequency(frequency_df):
    increase_df = pd.DateFrame(columns=["Week", "Date"])
    words = frequency_df.columns.tolist()[2:]
    for index, row in frequency_df.iterrows():
        if index != 0:
            for word in words:
                last_week_freq = frequency_df.iloc[index-1, word]
                this_week_freq = frequency_df.iloc[index, word]
                increase = (this_week_freq-last_week_freq)/last_week_freq
                increase_df[word] = increase
    top_dict = {}
    last_row = increase_df.tail(1)
    for word in last_row.columns:
        word_inc = last_row[0, word]
        if len(top_dict) > 10:
            top_dict[word] = word_inc
        else:
            lowest_in_top_dict_key = min(top_dict, key=top_dict.get)
            lowest_in_top_dict_val = top_dict[lowest_in_top_dict_key]
            if word_inc > lowest_in_top_dict_val:
                del top_dict[lowest_in_top_dict_key]
                top_dict[word] = increase
    trending_words = top_dict.keys().tolist()
    columns = ["Week", "Date"].extend(trending_words)
    trends_df = increase_df[columns]
    return trends_df


if __name__ == '__main__':
    print(get_start_and_end_date(2020, 11, 3))




