import pandas as pd
import numpy as np
import os
from pathlib import Path
from csv import reader
from scipy import stats
import matplotlib.pyplot as plt
import datetime
from src.make_results import get_df_filtered_on_industry_and_region
from src.make_results import get_region_list
from src.run import get_regions

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
    p = get_path_type_year("d", year)
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
    p = get_path_type_year("s", year)
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
    p = get_path_type_year("d", year)
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
    p = get_path_type_year("s", year)
    ind_part = "All ads" if len(industries) == 0 else "_".join(industries)
    reg_part = "all regions" if len(region_list) == 0 else "_".join(region_list)
    directory = "..\\data\\output"
    result_path = directory + "\\ads_per_week_{}_{}_{}.csv".format(year, ind_part, reg_part)
    already_calculated = result_path in [str(p) for p in sorted(Path(directory).iterdir())]
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
        date_list = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days + 1)]
        ads_per_week_df = pd.DataFrame(columns=["Week", "Date", "Ads"])
        for i in range(52):
            idx = range(i * 7, i * 7 + 7)
            week_dates = [date_list[i] for i in idx]
            start = np.datetime64(week_dates[0])
            end = np.datetime64(week_dates[-1])
            week_df = df[(df['Registrert dato'] >= start) & (df['Registrert dato'] <= end)]
            print("Week " + str(i + 1) + ", ads: " + str(len(week_df)))
            number_of_ads = len(week_df)
            row = {"Week": i + 1, "Date": start, "Ads": number_of_ads}
            ads_per_week_df = ads_per_week_df.append(row, ignore_index=True)
        ads_per_week_df.to_csv(result_path, sep=",", index=False)
    plt.bar(ads_per_week_df["Week"], ads_per_week_df["Ads"], label=ind_part + " in " + reg_part)
    plt.title("Struct. ads per week {}".format(year), pad=10)
    plt.ylabel("Amount", labelpad=8)
    plt.xlabel("Week No.")
    max_y = max(ads_per_week_df["Ads"])
    plt.ylim(0, max_y * 1.1)
    plt.tight_layout()
    plt.legend(loc="upper right")
    plot_path_fig = "..\\plots\\Numbers\\ads_per_week_{}_{}_{}.png".format(year, ind_part, reg_part)
    plt.savefig(plot_path_fig)
    plt.show()


def plot_word_amount_each_analysis_year(year):
    print("\nCalculating word count for ad descriptions in {}...".format(year))
    directory = "..\\data\\output"
    result_path = directory + "\\desc_word_count_{}.csv".format(year)
    already_calculated = result_path in [str(p) for p in sorted(Path(directory).iterdir())]
    if already_calculated:
        df = pd.read_csv(result_path, sep=",")
    else:
        p = get_path_type_year("d", year)
        df = pd.read_csv(p, sep=",", header=0)
        df = df.sample(n=10000, random_state=1)
        df['Antall ord'] = df['Stillingsbeskrivelse vasket'].str.split().str.len()
        df = df[["Stilling Id", "Antall ord"]]
        df.to_csv(result_path, sep=",", index=False)
    data = df["Antall ord"]
    mean = data.mean()
    params = stats.exponweib.fit(data, floc=0, f0=1)
    shape = params[1]
    scale = params[3]
    values, bins, hist = plt.hist(data, bins=40, range=(0, max(data)), density=True, label="Number of ads")
    center = (bins[:-1] + bins[1:]) / 2
    plt.plot(center, stats.exponweib.pdf(center, *params), lw=4, label='Weibell estimation')
    plt.xlabel("Number of Words", labelpad=8)
    plt.gca().yaxis.set_visible(False)
    plt.legend(loc="upper right")
    footnote = "Mean {}, shape {}, scale {}".format(round(mean, 1), round(shape, 2), round(scale, 1))
    plt.gcf().text(0.64, 0.78, footnote, fontsize=8)
    plt.title("Ad Description Word Count {}".format(year))
    plt.tight_layout()
    plot_path_fig = "..\\plots\\exploratory\\description_word_amount{}.png".format(year)
    plt.savefig(plot_path_fig)
    plt.show()


def plot_number_of_ads_hist_all_years():
    print("\nPlotting ad count each year..")
    directory = "..\\data\\output"
    result_path = directory + "\\number_of_ads_all_years_df.csv"
    already_calculated = result_path in [str(p) for p in sorted(Path(directory).iterdir())]
    if already_calculated:
        result_df = pd.read_csv(result_path, sep=",")
    else:
        result_df = pd.DataFrame(columns=["Year", "Struct", "Descript", "Both"])
        for year in range(2013, 2019):
            print("Calculating data for {}".format(year))
            s_path = get_path_type_year("s", year)
            d_path = get_path_type_year("d", year)
            s_df = pd.read_csv(s_path, sep=";", usecols=["Stilling id"])
            d_df = pd.read_csv(d_path, sep=",", usecols=["Stilling Id"])
            d_df = d_df.rename(columns={"Stilling Id": "Stilling id"})
            both_df = pd.merge(s_df, d_df, on="Stilling id", how="inner")
            row = {"Year": year, "Struct": len(s_df), "Descript": len(d_df), "Both": len(both_df)}
            result_df = result_df.append(row, ignore_index=True)
        result_df.to_csv(result_path, sep=",", index=False)
        print("Results save to file {}".format(result_path))
    labels = [year for year in range(2013, 2019)]
    div_factor = 1000
    struct_count = [row["Struct"] for id, row in result_df.iterrows()]
    struct_count = [amount // div_factor for amount in struct_count]
    descript_count = [row["Descript"] for id, row in result_df.iterrows()]
    descript_count = [amount // div_factor for amount in descript_count]
    both_count = [row["Both"] for id, row in result_df.iterrows()]
    both_count = [amount // div_factor for amount in both_count]
    bar_width = 0.25
    r1 = np.arange(len(labels))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    plt.bar(r1, struct_count, width=bar_width, edgecolor='white', label='Structured')
    plt.bar(r2, descript_count, width=bar_width, edgecolor='white', label='Descriptive')
    plt.bar(r3, both_count, width=bar_width, edgecolor='white', label='Struct. ∩ Descript.')
    plt.ylabel('Amount (x 1000)', labelpad=8)
    plt.xlabel('Year')
    plt.title('Number of ads 2013-2018')
    plt.xticks([r + bar_width for r in range(len(labels))], labels)
    plt.legend()
    plt.tight_layout()
    out_path_fig = "..\\plots\\exploratory\\number_of_ads_hist_all_years.png"
    plt.savefig(out_path_fig)
    plt.show()


def plot_hist_amount_of_ads_all_industries(year, region_list):
    print("\nCalculating ad count each industry for {}...".format(year))
    directory = "..\\data\\output"
    if len(region_list) == 0:
        result_path = directory + "\\struct_count_per_industry_{}.csv".format(year)
        label = "All regions"
        out_path = "..\\plots\\exploratory\\ads_per_industry_{}.png".format(year)
    else:
        result_path = directory + "\\struct_count_per_industry_{}_{}.csv".format(year, "_".join(region_list))
        label = ", ".join(region_list)
        out_path = "..\\plots\\exploratory\\ads_per_industry_{}_{}.png".format(year, "_".join(region_list))
    already_calculated = result_path in [str(p) for p in sorted(Path(directory).iterdir())]
    if already_calculated:
        df = pd.read_csv(result_path, sep=",")
    else:
        s_path = get_path_type_year("s", year)
        df = pd.read_csv(s_path, sep=";", usecols=["Stilling id", "Yrke grovgruppe", "Arbeidssted fylke"])
        regions = get_region_list(region_list)
        df = get_df_filtered_on_industry_and_region([], regions, df)
        df = df.groupby(['Yrke grovgruppe']).size().reset_index(name='counts')
        df.to_csv(result_path, sep=",", index=False)
    div_factor = 1000
    df["counts"] = [c / div_factor for c in df["counts"].tolist()]
    plt.bar(df["Yrke grovgruppe"], df["counts"], label=label)
    xticks = ["Akademia", "Barn", "Butikk", "Bygg", "Helse", "Inudstri", "Ingen data", "Ingeniør", "Primær", "Kontor",
              "Ledere", "Meglere", "Reise", "Service", "Lærere"]
    plt.xticks(range(len(df)), xticks, rotation=90)
    plt.title("Number of ads per industry in {}".format(year))
    plt.ylabel("Amount (x {})".format(div_factor), labelpad=8)
    plt.legend(loc="upper right")
    # for i, v in enumerate(df["counts"].tolist()):
    # plt.text(i, v+0.03, str(round(v, 1)), color="blue", fontweight='bold', size=8, ha='center')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    plt.close()


def plot_hist_amount_of_ads_all_regions(year):
    print("\nCalculating ad count each region for {}...".format(year))
    region_dict = get_regions()
    directory = "..\\data\\output"
    result_path = directory + "\\struct_count_per_region_{}.csv".format(year)
    already_calculated = result_path in [str(p) for p in sorted(Path(directory).iterdir())]
    if already_calculated:
        df = pd.read_csv(result_path, sep=",")
    else:
        s_path = get_path_type_year("s", year)
        df_all = pd.read_csv(s_path, sep=";", usecols=["Stilling id", "Arbeidssted fylke"])
        df_grouped = df_all.groupby(['Arbeidssted fylke']).size().reset_index(name='counts')
        df = pd.DataFrame()
        for region in region_dict.values():
            regions = get_region_list([region])
            if len(regions) > 1:
                regions.remove(region)
            ad_number = 0
            for r in regions:
                r_counts = int(df_grouped[df_grouped["Arbeidssted fylke"] == r]["counts"])
                ad_number += r_counts
            row = {'Arbeidssted fylke': region, "counts": ad_number}
            df = df.append(row, ignore_index=True)
        df.to_csv(result_path, sep=",", index=False)
    div_factor = 1000
    df["counts"] = [c / div_factor for c in df["counts"].tolist()]
    plt.barh(df["Arbeidssted fylke"], df["counts"], color="orange")
    plt.title("Number of ads per region in {}".format(year))
    plt.xlabel("Amount (x {})".format(div_factor), labelpad=8)
    yticks = ["Troms og Finm.", "Nordland", "Trøndelag", "Mør. og Roms.", "Vestland", "Rogaland", "Agder",
              "Vestf. og Tele.", "Viken", "Oslo", "Innlandet"]
    plt.yticks(range(len(df)), yticks)
    plt.xlim(0, max(df["counts"]) * 1.1)
    for i, v in enumerate(df["counts"].tolist()):
        plt.text(v + 0.4, i - 0.2, str(round(v, 1)), color="orange", fontweight='bold')
    plt.tight_layout()
    out_path = "..\\plots\\exploratory\\ads_per_region_{}.png".format(year)
    plt.savefig(out_path)
    plt.show()


def get_path_type_year(type, year):
    if type == "d":
        paths = descript_paths
    elif type == "s":
        paths = struct_paths
    p = ""
    for path in paths:
        p_list = str(path).split(sep="\\")
        if int(p_list[-1][0:4]) == year:
            p = path
    return p


if __name__ == '__main__':
    print("Running method ...")
    # view_numbers_of_ads_each_week(2017, ["Ingeniør- og ikt-fag"], ["Trøndelag"])
    # view_numbers_of_ads_each_week(2018, ["Butikk- og salgsarbeid"], ["Nordland"])
    # view_ads_count_from_to(2018, 2020)
    # view_all_struct_col_names(2020)
    # view_all_descript_col_names(2019)
    # view_n_first_descriptions_year(3, 2018)
    # verify_descript_relates_to_struct()
    # delete_empty_descript_rows()
    # plot_word_amount_each_analysis_year(2017)
    # plot_number_of_ads_hist_all_years()
    # plot_hist_amount_of_ads_all_industries(2017, region_list=[])
    # plot_hist_amount_of_ads_all_regions(2017)
