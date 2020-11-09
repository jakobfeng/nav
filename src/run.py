# run script. Three methods to choose from
import datetime
from src.make_results import create_results
from src.make_results import get_start_and_end_date
from src.make_results import create_path_for_results
from src.make_results import plot_word_cloud_from_dict
from src.make_results import plot_frequency_word_list


def make_word_clouds(industry_list, regions, start, end, classes):
    print("Making word cloud for " + ", ".join(industry_list) + " in " + ", ".join(regions) + "\n")
    path = create_path_for_results(industry_list, regions, start, end)
    results_df = create_results(industry_list, regions, start, end, classes, path)
    plot_word_cloud_from_dict(results_df, classes, industry_list, regions, start, end, path)


def make_frequency_plot(industry_list, regions, start, end, classes, word_list):
    path = create_path_for_results(industry_list, regions, start, end)
    results_df = create_results(industry_list, regions, start, end, classes, path)
    plot_frequency_word_list(results_df, industry_list, regions, start, end, classes, word_list, path)


def make_trend_plot(industry_list, regions, year, month, classes, months_lookback):
    start, end = get_start_and_end_date(year, month, months_lookback)
    path = create_path_for_results(industry_list, regions, start, end)
    results_df = create_results(industry_list, regions, start, end, classes, path)
    print(results_df)


def get_industries():
    industries_ = {0: "Ingen yrkesbakgrunn eller uoppgitt", 1: "Ledere", 2: "Ingeniør- og ikt-fag", 3: "Undervisning",
                  4: "Akademiske yrker", 5: "Helse, pleie og omsorg", 6: "Barne- og ungdomsarbeid",
                  7: "Meglere og konsulenter",
                  8: "Kontorarbeid", 9: "Butikk- og salgsarbeid", 10: "Jordbruk, skogbruk og fiske",
                  11: "Bygg og anlegg",
                  12: "Industriarbeid", 13: "Reiseliv og transport", 14: "Serviceyrker og annet arbeid"}
    return industries_


def get_regions():
    regions_ = {0: "Troms og Finnmark", 1: "Nordland", 2: "Trøndelag", 3: "Møre og Romsdal", 4: "Vestland",
                5: "Rogaland", 6: "Agder", 7: "Vestfold og Telemark", 8: "Viken", 9: "Oslo", 10: "Innlandet"}
    return regions_


if __name__ == '__main__':
    industries = get_industries()

    current_regions = get_regions()

    # -----------------------------------------------------------------------
    industry_ = [9]
    region_ = [2]
    start_date_ = datetime.date(2017, 1, 1)
    end_date_ = datetime.date(2017, 12, 31)
    classes_ = [1, 2, 3]
    word_list_ = ["selvstendig", "positiv", "team"]
    # _______________________________________________________________________
    region_input = [current_regions[i] for i in region_]
    for i in industries.keys():
        industry_input = [industries[i]]
        make_word_clouds(industry_input, region_input, start_date_, end_date_, classes_)
    # make_frequency_plot(industry_input, region_input, start_date_, end_date_, classes_, word_list_)
    # make_trend_plot(industry_input, region_input, end_date_.year, end_date_.month, classes_, 1)
