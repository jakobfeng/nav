# imports
from functools import reduce

from bs4 import BeautifulSoup
import re
from pathlib import Path
import pandas as pd
import time
from time import strftime
from time import gmtime
import functools


# function to remove HTML tags
def remove_html_tags(text, index):
    if len(text) > 0:
        clean = re.compile('<.*?><.*?>')
        text = re.sub(clean, '.', text)
        return BeautifulSoup(text, 'html.parser').get_text()
    else:
        print("Missing data from index {}: ".format(index))


def multiple_replace(string):
    cleaned_desc = string.translate({ord(c): "." for c in "!@#$%^*;<>\|`~-=_+¿"})
    rep_dict = {"[Klikk her]": "", "#": "", "\n": ".", "..": ".", ":.": ".", ", ,": ", ", "/": " ", "&": "og", "?": ".", ". . ": ".", " ca.": " ca"}
    pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict, key=len, reverse=True)]), flags=re.DOTALL)
    cleaned_desc = pattern.sub(lambda x: rep_dict[x.group(0)], cleaned_desc)
    cleaned_desc = re.sub(r'\.+', ".", cleaned_desc)  # only one period (not more after one another)
    cleaned_desc = re.sub(r'\.(?! )', '. ', re.sub(r' +', ' ', cleaned_desc))  # assure space after each period
    rep_dict = {"..": ". ", ":.": ". ", ", ,": ". ", ". . ": ". ", ".  . ": ". ", ". , . ,": ". ", ".,": ". ", ": .": ". ", ",. ": ". ", " . ": ". "}
    pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict, key=len, reverse=True)]), flags=re.DOTALL)
    cleaned_desc = pattern.sub(lambda x: rep_dict[x.group(0)], cleaned_desc)
    return cleaned_desc


def clean_descript_file(path, year, out_path_all):
    cleaned_files = sorted(Path("..\\data\\input\\descript_cl").iterdir())
    already_cleaned = False
    for p in cleaned_files:
        y = str(p)[-20:-16]
        if y == year:
            already_cleaned = True
    if not already_cleaned:
        start_time = time.time()
        print("\nCleaning descript year " + year + "\n")
        out_path_year = str(out_path_all + year + "_descript_cl.csv")
        df = pd.read_csv(path, header=0, sep=";", encoding="utf-8")
        col_name = df.columns[3]
        df = df[df[col_name].notna()]
        print("Cleaning {} rows".format(df[col_name].count()))
        for row in df.iterrows():
            index = int(row[0])
            if index % 1000 == 0:
                print(index)
            descript = row[1][3]
            clean_desc = remove_html_tags(descript, index)
            clean_desc = multiple_replace(clean_desc)
            df.loc[index, col_name] = clean_desc
        try:
            df.to_csv(path_or_buf=out_path_year, sep=",", index=False)
            print("\nCleaned descript data for " + str(year) + " saved to " + out_path)
        except Exception as e:
            print(e)
        finally:
            print("Time: " + str(strftime("%H:%M:%S", gmtime(time.time() - start_time))))


def clean_descript_files_all_years(path, out_path_all):
    paths = sorted(Path(path).iterdir())
    for p in paths:
        year = str(p)[-17:-13]
        print(year)
        clean_descript_file(p, year, out_path_all)


def replace_norwegian_characters():
    rep_dict = {"Ã¦": "æ", "Ã¥": "å", "Ã¸": "ø", "Ã˜": "Ø"}
    pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict, key=len, reverse=True)]),
                         flags=re.DOTALL)
    cleaned_files = sorted(Path("..\\data\\input\\descript_cl").iterdir())
    for descript in cleaned_files:
        if "descript_cl2" in str(descript):
            text = open(descript, "r")
            result = ""
            for line in text.readlines():
                new_line = pattern.sub(lambda x: rep_dict[x.group(0)], line)
                result += new_line
            x = open("..\\data\\input\\descript_cl\\2013_descript_cl3.csv", "w")
            x.writelines(result)
            x.close()


if __name__ == '__main__':
    descript_path = "..\\data\\input\\descript"
    out_path = "..\\data\\input\\descript_cl\\"
    clean_descript_files_all_years(descript_path, out_path)
    # replace_norwegian_characters()
