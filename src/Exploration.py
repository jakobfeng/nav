import pandas as pd
import numpy as np
import os
from pathlib import Path
from csv import reader
import matplotlib.pyplot as plt
import datetime
from src.make_results import get_df_filtered_on_industry_and_region
from src.make_results import get_region_list


struct_paths = sorted(Path("../data/input/struct").iterdir())  # list all structured datasets paths
descript_paths = sorted(Path("../data/input/descript_cl").iterdir())  # list all descriptive datasets paths
out_path = Path("../data/output/")
plot_path = Path("../plots/")


def view_ads_count_from_to(from_year, to_year):
    ads = {}
    for path in struct_paths:
        number_of_ads = 0
        number_of_vacancies = 0
        p_list = str(path).split(sep="\\")
        year = int(p_list[-1][0:4])
        if from_year <= year <= to_year:
            print(year)
            with open(path, 'r', encoding='utf-8') as read_obj:
                csv_reader = reader(read_obj, delimiter=';')
                for row in csv_reader:
                    if row[0] != '':  # header row
                        number_of_ads += 1
                        number_of_vacancies += int(row[28])
            ads[year] = [number_of_ads, number_of_vacancies]

    save_path = str(out_path + "numberOfAds_" + str(from_year) + "_" + str(to_year) + ".txt")
    file = open(save_path, "w")
    for key in ads.keys():
        y_string = (str(key) + ": " + str(ads[key][0]) + " ads, " + str(ads[key][1]) + " vacancies")
        file.writelines(y_string + "\n")
    file.close()
    print("\nValues saved to " + save_path)

    #  plot
    divisor = 1000
    ads_values, vac_values = zip(*ads.values())
    ads_values = [x / divisor for x in ads_values]
    vac_values = [x / divisor for x in vac_values]
    plt.plot(ads.keys(), ads_values, label="Ads")
    plt.plot(ads.keys(), vac_values, label="Vacancies")
    plt.legend()
    plt.title("Ads and Vacancies - " + str(from_year) + " to " + str(to_year))
    ax = plt.gca()
    ax.locator_params(integer=True)
    plt.xlabel("Year", labelpad=8, size=11)
    plt.ylabel("Amount (" + str(divisor) + ")", labelpad=8, size=11)
    plt.tight_layout()
    plot_path_fig = plot_path + "numberOfAds_" + str(from_year) + "_" + str(to_year) + ".png"
    plt.savefig(plot_path_fig)
    print("Plot saved to " + plot_path_fig)


def view_all_descript_col_names(year):
    p = ""
    for path in descript_paths:
        p_list = str(path).split(sep="\\")
        if int(p_list[-1][0:4]) == year:
            p = path

    with open(p, 'r') as read_obj:
        csv_reader = reader(read_obj, delimiter=",")
        header_string = next(csv_reader)
    save_path = str(out_path + "descript_col_indices_" + str(year) + ".txt")
    file = open(save_path, "w")
    file.writelines("Index\tColumn name" + "\n")
    for column in header_string:
        y_string = (str(header_string.index(column)) + "\t" + str(column))
        file.writelines(y_string + "\n")
    print("\nDescriptive column names and indices from {} have been saved to {}".format(year, save_path))
    file.close()


def view_all_struct_col_names(year):
    p = ""
    for path in struct_paths:
        p_list = str(path).split(sep="\\")
        if int(p_list[-1][0:4]) == year:
            p = path

    with open(p, 'r') as read_obj:
        csv_reader = reader(read_obj, delimiter=";")
        header_string = next(csv_reader)
    save_path = str(out_path + "struct_col_indices_" + str(year) + ".txt")
    file = open(save_path, "w")
    file.writelines("Index\tColumn name" + "\n")
    for column in header_string:
        y_string = (str(header_string.index(column)) + "\t" + str(column))
        file.writelines(y_string + "\n")
    print("\nStructured column names and indices from {} have been saved to {}".format(year, save_path))
    file.close()


def view_n_first_descriptions_year(n, year):
    print(str("\n--------View {} first job ads descriptions from {}----------\n").format(n, year))
    p = ""
    for path in descript_paths:
        p_list = str(path).split(sep="\\")
        if int(p_list[-1][:4]) == year:
            p = path
    df = pd.read_csv(p, header=0, sep=",", nrows=n)
    descriptions = {}
    for row in df.iterrows():
        index = row[0]
        id = row[1][1]
        title = row[1][2]
        desc = row[1][3]
        descriptions[index] = [id, title, desc]
    for key in descriptions.keys():
        print("Index: " + str(key), ", ID: " + str(descriptions[key][0]) + ", title: " + str(descriptions[key][1]))
        print("Description: \n")
        print(descriptions[key][2])
        print("---------------------------------------------------\n")


def verify_descript_relates_to_struct():
    conclusion = {}
    for d in descript_paths:
        number_of_ads = 0
        number_of_hits = 0
        p_list = str(d).split(sep="\\")
        d_year = p_list[-1][0:4]
        print("Checking descript_cl year " + d_year)
        s_path = ""
        for s in struct_paths:
            p_list = str(s).split(sep="\\")
            s_year = p_list[-1][0:4]
            if s_year == d_year:
                print("Found path struct year " + s_year)
                s_path = s  # path to struct the same year
                break
        descript_df = pd.read_csv(d, header=0, sep=",")
        struct_df = pd.read_csv(s_path, header=0, sep=";")
        print("Number of ads in descript_cl: {}, number of ads in struct: {}".format(len(descript_df), len(struct_df)))
        for d_row in descript_df.iterrows():
            number_of_ads += 1
            try:
                d_id = np.int64(d_row[1][1])
                s_row = struct_df.loc[struct_df["Stilling id"] == d_id]
                if not s_row.empty:
                    number_of_hits += 1
            except Exception as e:
                print(e)
                print(d_row)
        print("In year {}: {} ads and {} hits".format(d_year, number_of_ads, number_of_hits))
        conclusion[d_year] = [number_of_ads, number_of_hits]
    for year in conclusion.keys():
        ads = conclusion[year][0]
        hits = conclusion[year][1]
        print("{}: {} ads in descript_cl, {} of them found in struct".format(year, ads, hits))


def delete_empty_descript_rows():
    for d in descript_paths:
        p_list = str(d).split(sep="\\")
        d_year = p_list[-1][0:4]
        print("Deleting empty rows from year " + d_year)
        descript_df = pd.read_csv(d, header=0, sep=",")
        if descript_df.isnull().values.any():
            print("Found missing values in year " + d_year)
            is_NaN = descript_df.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = descript_df[row_has_NaN]
            print(rows_with_NaN)
            descript_df = descript_df.dropna()
            descript_df.to_csv(d, sep=",", index=False)


def view_numbers_of_ads_each_week(year, industries, region_list):
    regions = get_region_list(region_list)
    p = ""
    for path in struct_paths:
        p_list = str(path).split(sep="\\")
        if int(p_list[-1][0:4]) == year:
            p = path
    ind_part = "All ads" if len(industries) == 0 else "_".join(industries)
    reg_part = "all regions" if len(region_list) == 0 else "_".join(region_list)
    directory = "..\\data\\output"
    result_path = directory + "\\ads_per_week_{}_{}_{}.csv".format(year, ind_part, reg_part)
    already_calculated = result_path in [str(p) for p in sorted(Path(directory).iterdir())]
    already_calculated = False # REMOVE
    if already_calculated:
        ads_per_week_df = pd.read_csv(result_path)
    else:
        df = pd.read_csv(p, sep=";")
        df = get_df_filtered_on_industry_and_region(industries, regions, df)
        df.to_csv("..\\data\\output\\ads_2018_butikk_nordland_explo.csv", sep=",", index=False)
        date_format = "%Y-%m-%d"
        df['Registrert dato'] = pd.to_datetime(df['Registrert dato'], format=date_format)
        df['Registrert dato'] = df['Registrert dato'].dt.date
        start = datetime.date(year, 1, 1)
        end = datetime.date(year, 12, 31)
        date_list = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days+1)]
        ads_per_week_df = pd.DataFrame(columns=["Week", "Date", "Ads"])
        for i in range(52):
            idx = range(i * 7, i * 7 + 7)
            week_dates = [date_list[i] for i in idx]
            start = np.datetime64(week_dates[0])
            end = np.datetime64(week_dates[-1])
            week_df = df[(df['Registrert dato'] >= start) & (df['Registrert dato'] <= end)]
            print("Week " + str(i+1) + ", ads: " + str(len(week_df)))
            number_of_ads = len(week_df)
            row = {"Week": i+1, "Date": start, "Ads": number_of_ads}
            ads_per_week_df = ads_per_week_df.append(row, ignore_index=True)
        ads_per_week_df.to_csv(result_path, sep=",", index=False)
    plt.bar(ads_per_week_df["Week"], ads_per_week_df["Ads"], label=ind_part + " in " + reg_part)
    plt.title("Struct. ads per week {}".format(year), pad=10)
    plt.ylabel("Amount", labelpad=8)
    plt.xlabel("Week No.")
    max_y = max(ads_per_week_df["Ads"])
    plt.ylim(0, max_y*1.1)
    plt.tight_layout()
    plt.legend(loc="upper right")
    plot_path_fig = "..\\plots\\Numbers\\ads_per_week_{}_{}_{}.png".format(year, ind_part, reg_part)
    plt.savefig(plot_path_fig)
    plt.show()


if __name__ == '__main__':
    print("Running method ...")
    # view_numbers_of_ads_each_week(2017, ["Ingeniør- og ikt-fag"], ["Trøndelag"])
    view_numbers_of_ads_each_week(2018, ["Butikk- og salgsarbeid"], ["Nordland"])
    # view_ads_count_from_to(2018, 2020)
    # view_all_struct_col_names(2020)
    # view_all_descript_col_names(2019)
    # view_n_first_descriptions_year(3, 2018)
    # verify_descript_relates_to_struct()
    # delete_empty_descript_rows()

