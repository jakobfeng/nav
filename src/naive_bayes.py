# script for naive bayes models
import random
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from pathlib import Path
import _pickle as pickle
from ast import literal_eval

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

    number_of_hits = 0
    for index, row in outcome.iterrows():
        hit = row[1] == row[2]
        if hit:
            number_of_hits += 1
    accuracy = number_of_hits/len(outcome)
    accuracy_as_percent = round(accuracy*100, 1)
    print("Accuracy of Naive Bayes Model v.{} is: {}%".format(model_version, accuracy_as_percent))


if __name__ == '__main__':
    # training_path = "..\\data\\training\\training_set.csv"
    # training_df = pd.read_csv(training_path, sep=",")
    # train_naive_bayes_model(training_df)
    version = 1
    evaluate_model(version)
    # s_vector = "snill grei flink samarbeid kommunikasjon produktiv effektiv snill".split(" ")
    # model_ = load_model(path_)
    # label = get_class_naive_bayes(s_vector, model_)
