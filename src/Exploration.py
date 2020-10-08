import pandas as pd
import os
from pathlib import Path
from csv import reader
import matplotlib.pyplot as plt

struct_path = "..\\data\\input\\struct"
struct_paths = sorted(Path(struct_path).iterdir())  # list all structured datasets paths
descript_path = "..\\data\\input\\descript"
descript_paths = sorted(Path(descript_path).iterdir())  # list all descriptive datasets paths
out_path = "..\\data\\output\\"
plot_path = "..\\plots\\"


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
        csv_reader = reader(read_obj, delimiter=";")
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


def view_n_first_descritpions_year(n, year):
    print(str("\n--------View {} first job ads descriptions from {}----------\n").format(n, year))
    p = ""
    for path in descript_paths:
        p_list = str(path).split(sep="\\")
        if int(p_list[-1][:4]) == year:
            p = path
    p = "..\\data\\input\\2002_descript_clean.csv"
    df = pd.read_csv(p, header=0, sep=";", nrows=n)
    descriptions = {}
    for row in df.iterrows():
        index = row[0]
        print(row[1])
        assert False
        id = row[1][2]
        title = row[1][3]
        desc = row[1][4]
        descriptions[index] = [id, title, desc]
    for key in descriptions.keys():
        print("Index: " + str(key), ", ID: " + str(descriptions[key][0]) + ", title: " + str(descriptions[key][1]))
        print("Description: \n")
        print(descriptions[key][2])
        print("---------------------------------------------------\n")


if __name__ == '__main__':
    # view_ads_count_from_to(2018, 2020)
    # view_all_struct_col_names(2020)
    #view_all_descript_col_names(2019)
     view_n_first_descritpions_year(10, 2002)
