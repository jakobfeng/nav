from wordcloud import WordCloud
import matplotlib.pyplot as plt


def plot_frq_dict(freq_dict, label, result_path):
    replace_dict = {1: "Tasks", 2: "Traits", 3: "Requirements"}
    path = "..\\plots\\" + replace_dict[label] + "_" + result_path.split("\\")[-1][:-3] + "png"
    word_cloud = WordCloud(width=1100, height=800, max_words=1628, relative_scaling=1,
                           normalize_plurals=False).generate_from_frequencies(freq_dict)

    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(path)
    plt.close()
