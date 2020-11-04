import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessor import pre_process_sentence


#  label_lines_path = "..\\data\\training\\label_lines.csv"
labeled_lines_path = "..\\data\\training\\labeled_lines.csv"
training_set_out_path = "..\\data\\training\\training_set.csv"
test_set_out_path = "..\\data\\training\\test_set.csv"


def from_labeled_lines_to_training_and_test_set():
    preprocessed_df = pd.DataFrame(columns=['Vector', 'Label'])
    labeled_lines_df = pd.read_csv(labeled_lines_path, sep=",")
    line_index = labeled_lines_df.columns.get_loc("Setning")
    label_index = labeled_lines_df.columns.get_loc("Kategori")
    for line in labeled_lines_df.iterrows():
        sentence = line[1][line_index]
        vector = pre_process_sentence(sentence)
        if len(vector) > 0:
            label = line[1][label_index]
            preprocessed_row = {'Vector': vector, 'Label': label}
            preprocessed_df = preprocessed_df.append(preprocessed_row, ignore_index=True)
    training_set_df = preprocessed_df.sample(frac=0.8, random_state=200)  # random state is a seed value
    test_set_df = preprocessed_df.drop(training_set_df.index)
    training_set_df.to_csv(training_set_out_path, sep=",", index=False)
    print("Training set saved to " + training_set_out_path + "\n")
    test_set_df.to_csv(test_set_out_path, sep=",", index=False)
    print("Test set saved to " + test_set_out_path)


if __name__ == '__main__':
    from_labeled_lines_to_training_and_test_set()