import pandas as pd
import os
from pathlib import Path
from csv import reader
import matplotlib.pyplot as plt

path = "..//data//input//"
paths = sorted(Path(path).iterdir(), key=os.path.getmtime)  # list of paths to all datasets
out_path = "..//data//output//"
plot_path = "..//plots//"


def view_ads_count_from_to(from_year, to_year):
    ads = {}
    for path in paths:
        number_of_ads = 0
        number_of_vacancies = 0
        p_list = str(path).split(sep="/")
        year = int(p_list[-1][0:4])
        if from_year <= year <= to_year:
            print(year)
            with open(path, 'r', encoding='utf-8') as read_obj:
                csv_reader = reader(read_obj)
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
    ads_values = [x/divisor for x in ads_values]
    vac_values = [x/divisor for x in vac_values]
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


if __name__ == '__main__':
    view_ads_count_from_to(2017, 2020)
