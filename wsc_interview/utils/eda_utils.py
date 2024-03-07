import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from wsc_interview.bert import preprocess_text
from collections import Counter

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


def label_distribution(labels: list):
    """
    Calculate the label distribution of the dataset.
    :param labels: list of labels
    """
    return Counter(labels)


def text_length_distribution(texts: list):
    """
    Calculate the text length distribution of the dataset.
    :param texts: list of texts
    """
    return Counter([len(preprocess_text(text).split()) for text in texts])


def plot_text_length_distribution(texts: list, filename: str = None):
    """
    Plot the text length distribution of the dataset.
    :param texts: list of texts
    :param filename: filename to save the plot, if None the plot will be shown
    """
    text_len = [len(preprocess_text(text).split()) for text in texts]
    plot_histogram(text_len, title="Text Length Distribution", xlabel="Text Length", ylabel="Count", filename=filename)


def plot_histogram(data: dict, title: str = "", xlabel: str = "", ylabel: str = "", filename: str = None):
    """
    Plot a histogram of the data.
    :param data: dictionary of data {key: count}
    :param title: plot title
    :param xlabel: x axis label
    :param ylabel: y axis label
    :param filename: filename to save the plot, if None the plot will be shown
    """

    # plot histogram
    plt.hist(data, bins=max(data))

    # set plot labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # save or show plot
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def remove_stopwords(text: str):
    """
    Remove stopwords from the text (unink nltk stopwords).
    :param text: text to remove stopwords from
    :return: cleaned text
    """

    # remove stopwords
    output = " ".join([word for word in text.split() if word not in stopwords])
    return output


def word_frequency_distribution(texts: str):
    """
    Calculate the word frequency distribution of the dataset.
    :param texts: list of texts
    """
    words = [word for text in texts for word in remove_stopwords(preprocess_text(text)).split()]
    return Counter(words)


def word_label_corr(texts, labels, word, filename=None):
    """
    Calculate the word frequency distribution of the dataset.
    if the correlation coefficient is less than 0.5 the function will return None.
    otherwise, it will plot a heatmap of the correlation.

    :param texts: list of texts
    :param labels: list of labels
    :param word: word to check correlation with
    :param filename: filename to save the plot, if None the plot will be shown
    """

    # preprocess text and word
    word = remove_stopwords(preprocess_text(word))
    is_word = [word in remove_stopwords(preprocess_text(text)).split() for text in texts]

    # calculate correlation coefficient
    corr_coef = np.corrcoef(is_word, labels)[0, 1]

    if corr_coef < 0.5:
        return

    # plot confusion matrix
    pd.DataFrame({'text': texts, 'is_word': is_word, 'label': labels})
    cm = pd.DataFrame(confusion_matrix(labels, is_word), columns=[f'not {word}', word], index=['Neg', 'Pos'])
    sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues')
    plt.title(f"Confusion Matrix for word: {word} - Correlation Coefficient: {corr_coef}")

    # save or show plot
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def param_frequency_distribution(texts: list):
    """
    Calculate the word frequency distribution of the dataset.
    :param texts: list of texts
    """
    words = [word for text in texts for word in remove_stopwords(preprocess_text(text)).split()]
    return Counter(words)


def get_params_distribution(params: list, param_classes: list):
    """
    Calculate the parameters distribution of the dataset.
    :param params: list of parameters
    :param param_classes: list of parameters classes
    """

    freq = Counter(params)
    return dict([(k, freq.get(k, 0)) for k in param_classes])


def get_param_label_distribution(params: list, labels: list, param_classes: list):
    """
    Calculate the parameters distribution of the dataset.
    :param params: list of parameters
    :param labels: list of labels
    :param param_classes: list of parameters classes
    """
    freq = Counter(zip(params, labels))
    return {param: {0: freq.get((param, 0), 0),
                    1: freq.get((param, 1), 0)}
            for param in param_classes}


def bar_plot(data: dict, title="", xlabel="", ylabel="", filename: str = None,
             color: str = 'forestgreen', sort_by_values=False):
    """
    Plot a bar plot.
    :param data: data to plot - dictionary of x and y values
    :param title: plot title
    :param xlabel: x axis label
    :param ylabel: y axis label
    :param filename: filename to save the plot, if None the plot will be shown
    :param color: color of the bars
    :param sort_by_values: sort the data by values
    """

    # prepare data
    x = list(range(len(data)))
    x_ticks = sorted(data.keys(), key=data.get, reverse=True) if sort_by_values else list(sorted(data.keys()))
    y = [data[key] for key in x_ticks]

    # create plot
    plt.figure(figsize=(20, 13))
    plt.bar(x, y, color=color)
    plt.xticks(x, x_ticks, rotation=90)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # save or show plot
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_param_label_freq(data: dict, param_classes: list, filename: str = None):
    """
    Plot a multi bar plot of the parameters label distribution.
    :param data: dictionary of data {param: {label: count}}
    :param param_classes: list of parameters classes
    :param filename: filename to save the plot, if None the plot will be shown
    """

    # prepare data
    x = list(range(len(data)))
    x_ticks = sorted(param_classes)

    # create negative and positive values bars
    y_neg = [data.get(x).get(0, 0) for x in x_ticks]
    y_pos = [data.get(x).get(1, 0) for x in x_ticks]

    # create plot
    plt.figure(figsize=(20, 13))
    plt.bar(np.arange(len(x_ticks)) + 0.2, y_neg, 0.4, label='Neg', color='firebrick')
    plt.bar(np.arange(len(x_ticks)) - 0.2, y_pos, 0.4, label='Pos', color='forestgreen')

    # set plot labels
    plt.xticks(x, x_ticks, rotation=90)
    plt.title("Parameter Label Distribution")
    plt.xlabel("Parameter")
    plt.ylabel("Count")
    plt.legend()

    # save or show plot
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
