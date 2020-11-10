# script for naive bayes models
import random
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from pathlib import Path
import _pickle as pickle
from ast import literal_eval
import datetime
import matplotlib.pyplot as plt
import numpy as np

struct_paths = sorted(Path("../data/input/struct").iterdir())  # list all structured datasets paths
descript_paths = sorted(Path("../data/input/descript_cl").iterdir())  # list all descriptive datasets paths


# Training Model -------------------------------------------------------------------------------
def train_naive_bayes_model(training_set_df):
    category_0_list = training_set_df[training_set_df["Label"] == 0]["Vector"].tolist()
    category_1_list = training_set_df[training_set_df["Label"] == 1]["Vector"].tolist()
    category_2_list = training_set_df[training_set_df["Label"] == 2]["Vector"].tolist()
    category_3_list = training_set_df[training_set_df["Label"] == 3]["Vector"].tolist()

    # word frequency for each category
    freq_0 = get_frequency_dict(category_0_list)
    freq_1 = get_frequency_dict(category_1_list)
    freq_2 = get_frequency_dict(category_2_list)
    freq_3 = get_frequency_dict(category_3_list)
    frequencies = [freq_0, freq_1, freq_2, freq_3]

    # priors (number os sentences in each category)
    prior_0 = len(category_0_list) / len(training_set_df)
    prior_1 = len(category_1_list) / len(training_set_df)
    prior_2 = len(category_2_list) / len(training_set_df)
    prior_3 = len(category_3_list) / len(training_set_df)
    priors = {0: prior_0, 1: prior_1, 2: prior_2, 3: prior_3}

    # Laplace smoothing
    total_features_0 = len(freq_0)
    total_features_1 = len(freq_1)
    total_features_2 = len(freq_2)
    total_features_3 = len(freq_3)
    total_features = get_total_features(training_set_df)
    features = {0: total_features_0, 1: total_features_1, 2: total_features_2, 3: total_features_3, 4: total_features}

    directory = "..\\models"
    save_model(frequencies, priors, features, directory)


# Helping method to save models after training
def save_model(frequencies, priors, features, directory):
    existing_paths = sorted(Path(directory).iterdir())
    version = 1
    for p in existing_paths:
        if "naive_bayes" in str(p):
            version += 1
    model_path = directory + "\\naive_bayes_v{}.json".format(version)
    list_of_dicts = frequencies + [priors] + [features]
    with open(model_path, 'wb') as fd:
        fd.write(pickle.dumps(list_of_dicts))
    print("Model saved to {}".format(model_path))


# Helping method for training
def get_frequency_dict(category_sentence_list):
    counter = CountVectorizer()
    count = counter.fit_transform(category_sentence_list)
    word_list = counter.get_feature_names()
    count_list = count.toarray().sum(axis=0)
    freq = dict(zip(word_list, count_list))
    return freq


# Helping method in training
def get_total_features(training_set_df):
    docs = [row['Vector'] for index, row in training_set_df.iterrows()]
    vec = CountVectorizer()
    vec.fit_transform(docs)
    total_features = len(vec.get_feature_names())
    return total_features


# Running Model -------------------------------------------------------------------------------
def get_class_naive_bayes(vector, model):
    freq = model[0]
    prior = model[1]
    feature = model[2]

    likelihood_0 = get_probability_class_words(vector, freq[0], feature[0], feature[4])
    likelihood_1 = get_probability_class_words(vector, freq[1], feature[1], feature[4])
    likelihood_2 = get_probability_class_words(vector, freq[2], feature[2], feature[4])
    likelihood_3 = get_probability_class_words(vector, freq[3], feature[3], feature[4])

    posterior_0 = get_posterior(likelihood_0, prior[0])
    posterior_1 = get_posterior(likelihood_1, prior[1])
    posterior_2 = get_posterior(likelihood_2, prior[2])
    posterior_3 = get_posterior(likelihood_3, prior[3])

    sum_posteriors = posterior_0 + posterior_1 + posterior_2 + posterior_3

    norm_post_0 = posterior_0 / sum_posteriors
    norm_post_1 = posterior_1 / sum_posteriors
    norm_post_2 = posterior_2 / sum_posteriors
    norm_post_3 = posterior_3 / sum_posteriors

    norm_posteriors = [norm_post_0, norm_post_1, norm_post_2, norm_post_3]
    max_norm_post = max(norm_posteriors)
    class_label = norm_posteriors.index(max_norm_post)
    return class_label


# Helping method to load model when running classification algorithm
def load_model(model_version):
    path = "..\\models\\naive_bayes_v{}.json".format(model_version)
    with open(path, 'rb') as fd:
        list_of_dicts = pickle.load(fd)
    freq = [list_of_dicts[0], list_of_dicts[1], list_of_dicts[2], list_of_dicts[3]]
    priors = list_of_dicts[4]
    features = list_of_dicts[5]
    return [freq, priors, features]


# Helping method when running method on new sentence
def get_probability_class_words(vector, freq, total_category_features, total_features):
    prob_with_ls = []
    for word in vector:
        if word in freq.keys():
            count = freq[word]
        else:
            count = 0
        prob_with_ls.append((count + 1) / (total_category_features + total_features))
    word_probs = dict(zip(vector, prob_with_ls))
    acc_prob = 1
    for word in vector:
        acc_prob *= word_probs[word]
    return acc_prob


# Helping method when running the models on a new sentence
def get_posterior(likelihood, prior):
    return likelihood * prior


# Method for evaluating model
def evaluate_model(model_version):
    outcome = outcome_from_test_set(model_version)
    number_of_hits = 0
    for index, row in outcome.iterrows():
        hit = row[1] == row[2]
        if hit:
            number_of_hits += 1
    accuracy = number_of_hits/len(outcome)
    accuracy_as_percent = round(accuracy*100, 1)
    print("Accuracy of Naive Bayes Model v.{} is: {}%".format(model_version, accuracy_as_percent))


def outcome_from_test_set(model_version):
    testing_path = "..\\data\\training\\test_set.csv"
    testing_df = pd.read_csv(testing_path, sep=",")
    testing_df = testing_df.dropna()
    outcome = pd.DataFrame(columns=["Vector", "Label", "Output"])
    model = load_model(model_version)
    for index, row in testing_df.iterrows():
        vector = literal_eval(row[0])
        label = row[1]
        output = get_class_naive_bayes(vector, model)
        outcome_row = {"Vector": vector, "Label": label, "Output": output}
        outcome = outcome.append(outcome_row, ignore_index=True)
    return outcome


def explore_model(model_version, year):
    from src.Ad_classifier import classify_ad
    from src.Exploration import get_path_type_year
    directory = "..\\data\\output"
    result_path_ads = directory + "\\explore_model_ads_{}.csv".format(year)
    already_obtained_ads = result_path_ads in [str(p) for p in sorted(Path(directory).iterdir())]
    if already_obtained_ads:
        df_ads = pd.read_csv(result_path_ads, sep=",")
    else:
        s_path = get_path_type_year("s", year)
        s_df = pd.read_csv(s_path, sep=";", usecols=["Stilling id", 'Registrert dato', "Arbeidssted fylke", "Yrke grovgruppe"])
        d_path = get_path_type_year("d", year)
        d_df = pd.read_csv(d_path, sep=",", usecols=["Stilling Id", "Stillingsbeskrivelse vasket"])
        d_df = d_df.rename(columns={"Stilling Id": "Stilling id"})
        df = pd.merge(s_df, d_df, on="Stilling id", how="inner")
        df = df.sample(n=1000, random_state=1)
        df = df.reset_index()
        df.to_csv(result_path_ads, index=False, sep=",")
    result_path_lines = directory + "\\explore_model_lines_{}.csv".format(year)
    already_obtained_lines = result_path_lines in [str(p) for p in sorted(Path(directory).iterdir())]
    result_path_time_consumption = directory + "\\explore_model_time_{}.csv".format(year)
    if already_obtained_lines:
        df_lines = pd.read_csv(result_path_lines, sep=",")
        df_time_consumption = pd.read_csv(result_path_time_consumption, sep=",")
    else:
        df_time_consumption = pd.DataFrame(columns=["Number", "Time"])
        df_lines = pd.DataFrame(
            columns=['Stilling id', 'Registrert dato', 'Yrke grovgruppe', 'Setning', 'Pros. Setning', 'Kategori'])
        model = load_model(model_version)
        start_time = datetime.datetime.now()
        for index, ad in df_ads.iterrows():
            print("Classify ad no. :" + str(index + 1))
            labeled_ad = classify_ad(model, ad)
            df_lines = df_lines.append(labeled_ad, ignore_index=True)
            end_time = datetime.datetime.now()
            time_diff = (end_time - start_time)
            execution_time = time_diff.total_seconds()
            time_row = {"Number": index+1, "Time": execution_time}
            df_time_consumption = df_time_consumption.append(time_row, ignore_index=True)
        df_lines = df_lines.sort_values(by=["Kategori", "Registrert dato"])
        df_lines.to_csv(result_path_lines, index=False, sep=",")
        df_time_consumption.to_csv(result_path_time_consumption, sep=",", index=False)
    plot_dir = "..\\plots\\model"
    path_out_time = plot_dir + "\\time_consumption_{}.png".format(year)
    already_plotted_time = path_out_time in [str(p) for p in sorted(Path(plot_dir).iterdir())]
    if not already_plotted_time:
        plt.plot(df_time_consumption["Number"], df_time_consumption["Time"], linewidth=3, label="Accumulated Time")
        plt.title("Naive Bayes Classifier Time Consumption", pad=8)
        plt.xlabel("Ad Input Size")
        plt.ylabel("Time [s]")
        plt.legend(loc="upper left")
        time_used = df_time_consumption.iloc[len(df_time_consumption)-1]["Time"]
        avg_time = round(time_used/len(df_time_consumption), 3)
        footnote = "Mean time/ad: {} sec.".format(avg_time)
        plt.gcf().text(0.13, 0.8, footnote, fontsize=9)
        plt.tight_layout()
        plt.savefig(path_out_time)
        plt.show()
        plt.close()
    path_out_label_plot = plot_dir + "\\label_distribution_{}.png".format(year)
    already_plotted_label_dist = path_out_label_plot in [str(p) for p in sorted(Path(plot_dir).iterdir())]
    if not already_plotted_label_dist:
        label = "1000 random ads from {}".format(year)
        label_column = "Kategori"
        plot_bar_percent(df_lines, path_out_label_plot, label, label_column)


# Helping method for plotting category bar chart with percent
def plot_bar_percent(df_lines, path_out_label_plot, label, label_column):
    number_of_lines = len(df_lines)
    df_lines = df_lines.groupby([label_column]).size().reset_index(name='counts')
    df_lines["counts"] = [(c * 100) / number_of_lines for c in df_lines["counts"]]
    plt.bar(df_lines[label_column], df_lines["counts"], label=label)
    xticks = ["None", "Task", "Trait", "Requirement"]
    plt.xticks(range(len(xticks)), xticks)
    plt.xlabel("Category")
    plt.ylabel("Percent [%]", labelpad=8)
    plt.title("Naive Bayes Category Distribution", pad=8)
    plt.legend(loc="upper right")
    plt.ylim(0, max(df_lines["counts"].tolist()) * 1.08)
    for i, v in enumerate(df_lines["counts"].tolist()):
        plt.text(i, v + 1, str(round(v, 1)), color="C0", fontweight='bold', size=9, ha='center')
    plt.savefig(path_out_label_plot)
    plt.show()
    print(df_lines)


def explore_training_set():
    training_path = "..\\data\\training\\training_set.csv"
    training_df = pd.read_csv(training_path, sep=",")
    label = "Training set lines"
    path_out_label_plot = "..\\plots\\model\\label_distribution_training.png"
    label_column = "Label"
    plot_bar_percent(training_df, path_out_label_plot, label, label_column)


def explore_test_set(model_version):
    directory = "..\\data\\output"
    result_path_test_set = directory + "\\test_set_results.csv"
    already_in_file = result_path_test_set in [str(p) for p in sorted(Path(directory).iterdir())]
    if already_in_file:
        outcome_df = pd.read_csv(result_path_test_set, sep=",")
    else:
        outcome_df = outcome_from_test_set(model_version)
        outcome_df.to_csv(result_path_test_set, sep=",", index=False)
    categories = [0, 1, 2, 3]
    cat_names_dict = {0: "None", 1: "Task", 2: "Trait", 3: "Requirement"}
    count_df = pd.DataFrame(columns=["True Label"] +categories)
    for cat in categories:
        cat_row = {"True Label": cat_names_dict[cat]}
        cat_dict = outcome_df[outcome_df["Label"] == cat]  # true labels
        cat_group = cat_dict.groupby(by=["Output"]).size().reset_index(name='counts')  # grouped by output
        for c in categories:
            hit = cat_group[cat_group["Output"] == c]["counts"]
            cat_row[c] = 0 if len(hit) == 0 else int(hit)
        count_df = count_df.append(cat_row, ignore_index=True)
    print(count_df)
    true_none = count_df.iloc[0].tolist()[1:]
    true_none_sum = sum(true_none)
    true_1 = count_df.iloc[1].tolist()[1:]
    true_1_sum = sum(true_1)
    true_2 = count_df.iloc[2].tolist()[1:]
    true_2_sum = sum(true_2)
    true_3 = count_df.iloc[3].tolist()[1:]
    true_3_sum = sum(true_3)
    scalars = [true_none_sum, true_1_sum, true_2_sum, true_3_sum]
    labeled_none = count_df[0].tolist()
    labeled_1 = count_df[1].tolist()
    labeled_2 = count_df[2].tolist()
    labeled_3= count_df[3].tolist()
    labeled_none_scaled = [labeled_none[i]/scalars[i] for i in range(len(categories))]
    labeled_1_scaled = [labeled_1[i]/scalars[i] for i in range(len(categories))]
    labeled_2_scaled = [labeled_2[i]/scalars[i] for i in range(len(categories))]
    labeled_3_scaled = [labeled_3[i]/scalars[i] for i in range(len(categories))]
    bottom_2 = np.add(labeled_none_scaled, labeled_1_scaled).tolist()
    bottom_3 = np.add(bottom_2, labeled_2_scaled).tolist()
    bar_width = 0.5
    x_val = range(len(categories))
    plt.bar(x_val, labeled_none_scaled, width=bar_width, edgecolor='white', label='None')
    plt.bar(x_val, labeled_1_scaled, width=bar_width, edgecolor='white', label='Task', bottom=labeled_none_scaled)
    plt.bar(x_val, labeled_2_scaled, width=bar_width, edgecolor='white', label='Trait', bottom=bottom_2)
    plt.bar(x_val, labeled_3_scaled, width=bar_width, edgecolor='white', label='Req.', bottom=bottom_3)
    plt.title("Naive Bayes Test Set Analysis", pad=16)
    plt.ylabel("Proportion", labelpad=8)
    plt.xlabel("True Label")
    plt.ylim(0, 1.25)
    plt.xticks(range(len(categories)), [cat_names_dict[i] for i in range(len(categories))], fontweight='bold')
    plt.legend(loc="center", ncol=4, bbox_to_anchor=[0.64, 0.95], shadow=True, fancybox=True)
    footnote = "Model Label"
    plt.gcf().text(0.21, 0.84, footnote, fontsize=10)
    hit_none = labeled_none[0]/true_none_sum
    hit_1 = labeled_1[1]/true_1_sum
    hit_2 = labeled_2[2]/true_2_sum
    hit_3 = labeled_3[3]/true_3_sum
    hits = [hit_none, hit_1, hit_2, hit_3]
    colors = ["C0", "C1", "C2", "C3"]
    for i, v in enumerate(hits):
        plt.text(i, 1.023, str(round(v, 2)), color=colors[i], fontweight='bold', size=10, ha='center')
    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), colors):
        ticklabel.set_color(tickcolor)
    out_path = "..\\plots\\model\\test_set_label_analysis.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # training_path = "..\\data\\training\\training_set.csv"
    # training_df = pd.read_csv(training_path, sep=",")
    # train_naive_bayes_model(training_df)
    version = 1
    # explore_model(version, 2017)
    # explore_training_set()
    explore_test_set(version)
    # evaluate_model(version)
    # s_vector = "snill grei flink samarbeid kommunikasjon produktiv effektiv snill".split(" ")
    # model_ = load_model(path_)
    # label = get_class_naive_bayes(s_vector, model_)
