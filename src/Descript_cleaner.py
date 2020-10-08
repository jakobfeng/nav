# imports
from bs4 import BeautifulSoup
from csv import reader
import pandas as pd
import os

missing_data = {}

# function to remove HTML tags
def remove_html_tags(text, index, year):
    if len(text) > 0:
        return BeautifulSoup(text, 'html.parser').get_text()
    else:
        print("Missing data from index {}: ".format(index))
        if year not in missing_data.keys():
            missing_data[year] = [index]
        else:
            missing_data[year].append(index)


def clean_descript_file(path, year):
    out_path = path[:-4] + "_clean.csv"
    print(path)
    df = pd.read_csv(path, header=0, sep=";")
    col_name = df.columns[3]
    df = df[df[col_name].notna()]
    print("Cleaning {} rows".format(df[col_name].count()))
    for row in df.iterrows():
        index = int(row[0])
        if index % 50 == 0:
            print(index)
        descript = row[1][3]
        clean_desc = remove_html_tags(descript, index, year)
        clean_desc = clean_desc.replace("#", "")
        clean_desc = os.linesep.join([s for s in clean_desc.splitlines() if s])
        df.loc[index, col_name] = clean_desc
    df.to_csv(path_or_buf=out_path, sep=";")


if __name__ == '__main__':
    path = "..\\data\\input\\descript\\2002_descript.csv"
    year = 2002
    clean_descript_file(path, year)
