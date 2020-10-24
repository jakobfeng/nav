import pandas as pd
from pathlib import Path
import random
from nltk import tokenize
import re

# nltk.download('punkt')

struct_path = "..\\data\\input\\struct"
struct_paths = sorted(Path(struct_path).iterdir())  # list all structured datasets paths
descript_path = "..\\data\\input\\descript_cl"
descript_paths = sorted(Path(descript_path).iterdir())  # list all descriptive datasets paths
training_ads_path = "..\\data\\training\\training_ads.csv"
training_lines_path = "..\\data\\training\\training_lines.csv"

job_groups = ["Ledere", "Ingeniør- og ikt-fag", "Undervisning", "Akademiske yrker", "Helse, pleie og omsorg",
              "Barne- og ungdomsarbeid", "Meglere og konsulenter", "Kontorarbeid", "Butikk- og salgsarbeid",
              "Jordbruk, skogbruk og fiske", "Bygg og anlegg", "Industriarbeid", "Reiseliv og transport",
              "Serviceyrker og annet arbeid", "Ingen yrkesbakgrunn eller uoppgitt"]


def check_if_year_is_finished(rows_year, n):
    for group in job_groups:
        if group != "Ingen yrkesbakgrunn eller uoppgitt":
            number_of_rows = 0
            for row in rows_year:
                if row["Yrke grovgruppe"] == group:
                    number_of_rows += 1
            if number_of_rows < n:
                return False
    return True


def check_if_stilling_id_already_used(stilling_id, rows_year):
    for row_dict in rows_year:
        row_stilling_d = row_dict["Stilling id"]
        if row_stilling_d == stilling_id:
            return True
    return False


def make_ads_set(n):
    print("Finding " + str(n) + " random ads for each 14 job groups, every year from 2013 to 2018..\n")
    training_ads_df = pd.DataFrame(columns=['Stilling id', 'Registrert dato', 'Yrke grovgruppe',
                                            'Stillingsbeskrivelse vasket'])
    for descript in descript_paths:  # each year...
        job_group_count = {}
        for job in job_groups:
            job_group_count[job] = 0
        rows_year = []  # list of dictionaries, where each dict is one row to be added from this year
        descript_df = pd.read_csv(descript, header=0, sep=",")
        d_year = str(descript).split(sep="\\")[-1][0:4]
        print("Retrieving ads from year " + d_year + "...")
        for s in struct_paths:
            s_year = str(s).split(sep="\\")[-1][0:4]
            if s_year == d_year:
                s_path = s  # path to struct the same year
                break
        struct_df = pd.read_csv(s_path, header=0, sep=";")
        job_group_index = struct_df.columns.get_loc("Yrke grovgruppe")  # 17
        ad_date_index = struct_df.columns.get_loc("Registrert dato")  # 5
        is_finished = False
        while not is_finished:
            random_index = random.randint(0, len(descript_df) - 1)
            descript_row = descript_df.iloc[random_index]
            stilling_id = descript_row[1]
            description = descript_row[3]
            stilling_id_already_in_list = check_if_stilling_id_already_used(stilling_id, rows_year)
            if not stilling_id_already_in_list:
                struct_row = struct_df.loc[struct_df["Stilling id"] == stilling_id]
                if not struct_row.empty:
                    current_group = ""
                    job_group = struct_row.iloc[0, job_group_index]
                    for key in job_group_count.keys():
                        if key in job_group:
                            current_group = key
                            break
                    if current_group != "":
                        if job_group_count[current_group] < n:
                            ad_date = struct_row.iloc[0, ad_date_index]
                            row = {"Stilling id": stilling_id, 'Registrert dato': ad_date,
                                   "Yrke grovgruppe": current_group, "Stillingsbeskrivelse vasket": description}
                            rows_year.append(row)
                            job_group_count[current_group] += 1

            is_finished = check_if_year_is_finished(rows_year, n)
        for row in rows_year:
            training_ads_df = training_ads_df.append(row, ignore_index=True)
    training_ads_df.to_csv(training_ads_path, index=False, sep=";")
    print("\nTraining ads saved to " + training_ads_path)


def extend_training_set():
    print("\nSplitting all ads into individual lines...\n")
    training_lines_df = pd.DataFrame(columns=['Stilling id', 'Registrert dato', 'Yrke grovgruppe',
                                              'Setning', 'Kategori'])

    training_ads_df = pd.read_csv(training_ads_path, header=0, sep=";")
    for row in training_ads_df.iterrows():
        stilling_id = row[1][0]
        ad_date = row[1][1]
        job_group = row[1][2]
        description = row[1][3]
        sentences = tokenize.sent_tokenize(description)
        rows_of_lines = []
        for s in sentences:
            if len(s.split()) > 2:  # remove sentences shorter than three words
                s = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+
                |(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ",
                           s)  # remove links
                s = ' '.join([item for item in s.split() if '@' not in item])  # remove emails
                s = re.sub(r'\d+', '', s)  # remove numbers
                s = re.sub(r"\W+|_", " ", s)  # remove special characters
                line_dict = {"Stilling id": stilling_id, 'Registrert dato': ad_date,
                       "Yrke grovgruppe": job_group, "Setning": s, "Kategori": None}
                isempty = False
                if s == " " or s == "" or s == "." or s == " . " or s == ". ":
                    isempty = True
                if not isempty:
                    rows_of_lines.append(line_dict)
        for line in rows_of_lines:
            training_lines_df = training_lines_df.append(line, ignore_index=True)
    training_lines_df.to_csv(training_lines_path, index=False, sep=";")
    print("Training lines saved to " + training_lines_path)

if __name__ == '__main__':
    make_ads_set(2)
    extend_training_set()
