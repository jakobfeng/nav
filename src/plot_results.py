from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib as mpl
from matplotlib.dates import DateFormatter


def plot_frq_dict(freq_dict, label, industry_list, regions, start, end, directory):
    color_maps = {1: mpl.cm.Reds, 2: mpl.cm.Blues, 3: mpl.cm.Greens}
    main_color = color_maps[label](np.linspace(0, 1, 20))
    col_map = mpl.colors.ListedColormap(main_color[16:, :-1])
    mask = np.array(Image.open("../plots/cloud.png"))
    word_cloud = WordCloud(width=1300, height=950, max_words=100, relative_scaling=1, background_color="ivory",
                           normalize_plurals=False, prefer_horizontal=True, colormap=col_map,
                           mask=mask).generate_from_frequencies(freq_dict)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    label_to_title_dict = {1: "Tasks", 2: "Traits", 3: "Reqs."}
    title = label_to_title_dict[label] + " for " + ", ".join(industry_list).lower() + " in " + ", ".join(regions)
    if (len(industry_list) > 13) and (len(regions) > 10):
        title = label_to_title_dict[label] + " for all industries nationally"
    plt.title(title)
    start_str = start.strftime("%d.%b %y")
    end_str = end.strftime("%d.%b %y")
    footnote = start_str + " - " + end_str
    plt.gcf().text(0.73, 0.05, footnote, fontsize=8)
    plt.tight_layout()
    replace_dict = {1: "Tasks", 2: "Traits", 3: "Requirements"}
    path = directory + replace_dict[label] + ".png"
    plt.savefig(path)
    plt.close()


def plot_frequency_df(frequency_df, industries, classes, word_list, regions, directory):
    for word in word_list:
        plt.plot(frequency_df["Date"], frequency_df[word], label=word.capitalize())
    label_to_title_dict = {1: "Task", 2: "Trait", 3: "Req."}
    classes_part = [label_to_title_dict[k] for k in classes]
    classes_part = ", ".join(classes_part)
    if len(classes) == 3:
        classes_part = "Total"
    title = classes_part + " freq. for " + ", ".join(industries).lower() + " in " + ", ".join(regions)
    plt.title(title, pad=14, size=12)
    plt.legend(loc='upper right', ncol=1, fancybox=True, shadow=True)
    date_form = DateFormatter("%d.%m.%y")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(date_form)
    plt.xlabel("Date", size=11)
    plt.ylabel("Frequency", labelpad=8, size=11)
    print(frequency_df)
    path = directory + "_".join(word_list) + ".png"
    plt.savefig(path)
    plt.close()


def plot_trends_df(trend_df, industries, classes, trending_words, regions):
    for word in trending_words:
        plt.plot(trend_df["Date"], trend_df[word], label=word.capitalize())
    label_to_title_dict = {1: "Task", 2: "Trait", 3: "Req."}
    classes_part = [label_to_title_dict[k] for k in classes]
    classes_part = ", ".join(classes_part)
    if len(classes) == 3:
        classes_part = "Total"
    title = classes_part + " trends for " + ", ".join(industries).lower() + " in " + ", ".join(regions)
    plt.title(title, pad=14, size=12)
    plt.legend(loc='upper right', ncol=1, fancybox=True, shadow=True)
    date_form = DateFormatter("%d-%m")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(date_form)
    plt.xlabel("Date", size=11)
    plt.ylabel("Increase", labelpad=8, size=11)
    print(trend_df)
    path = "..\\plots\\trend" + ti
    tle + str(trend_df.head(1).iloc[0, 0]) + "_" + str(trend_df.tail(1).iloc[0, 0]) + ".png"
    plt.savefig(path)
    plt.close()