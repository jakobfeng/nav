# imports
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
import time
from time import strftime
from time import gmtime


# function to remove HTML tags
def remove_html_tags(text, index):
    if len(text) > 0:
        return BeautifulSoup(text, 'html.parser').get_text()
    else:
        print("Missing data from index {}: ".format(index))


def clean_descript_file(path, year):
    cleaned_files = sorted(Path("..\\data\\input\\descript_cl").iterdir())
    already_cleaned = False
    for p in cleaned_files:
        y = str(p)[-20:-16]
        if y == year:
            already_cleaned = True
    if not already_cleaned:
        start_time = time.time()
        print("\nCleaning descript year " + year + "\n")
        out_path = str("..\\data\\input\\descript_cl\\" + year + "_descript_cl.csv")
        df = pd.read_csv(path, header=0, sep=";")
        col_name = df.columns[3]
        df = df[df[col_name].notna()]
        print("Cleaning {} rows".format(df[col_name].count()))
        for row in df.iterrows():
            index = int(row[0])
            if index % 1000 == 0:
                print(index)
            descript = row[1][3]
            clean_desc = remove_html_tags(descript, index)
            clean_desc = clean_desc.replace("#", "")
            clean_desc = "".join([s for s in clean_desc.splitlines(True) if s.strip("\r\n")])
            df.loc[index, col_name] = clean_desc
        try:
            df.to_csv(path_or_buf=out_path, sep=",", index=False)
            print("\nCleaned descript data for " + str(year) + " saved to " + out_path)
        except Exception as e:
            print(e)
        finally:
            print("Time: " + str(strftime("%H:%M:%S", gmtime(time.time() - start_time))))


def clean_descript_files_all_years():
    path = "..\\data\\input\\descript"
    paths = sorted(Path(path).iterdir())
    for p in paths:
        year = str(p)[-17:-13]
        clean_descript_file(p, year)


if __name__ == '__main__':
    clean_descript_files_all_years()
