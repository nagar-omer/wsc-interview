import argparse
from wsc_interview.models.data_loaders import ActionDataset
from wsc_interview.models.bert import get_bert_uncased_tokenizer
from wsc_interview.utils.eda_utils import label_distribution, bar_plot, text_length_distribution, \
    plot_text_length_distribution, word_frequency_distribution, plot_histogram, word_label_corr, \
    get_params_distribution, get_param_label_distribution, plot_param_label_freq
import numpy as np


def run(data_path, params_file):
    # load the dataset (tokenizer is used for text preprocessing)
    ds = ActionDataset(data_path, params_file, tokenizer=get_bert_uncased_tokenizer())

    print("Number of instances: ", len(ds))
    print("Number of dropped instances: ", len(ds.dropped_instances))

    transcriptions, params, labels = ds.all_instances
    param_classes = ds.params_classes

    # label distribution
    ld = label_distribution(labels)
    print("Label Distribution: ", ld)
    bar_plot(ld, title="Label Distribution", xlabel="Label", ylabel="Count") #, filename="label_distribution.png")

    # text length distribution
    text_len = text_length_distribution(transcriptions)
    plot_text_length_distribution(transcriptions) #, filename="text_length_distribution.png")
    print("Text Length Mean: ", np.mean(list(text_len.keys())))
    print("Text Length Std: ", np.std(list(text_len.keys())))

    # word frequency distribution
    word_freq = word_frequency_distribution(transcriptions)
    plot_histogram(list(word_freq.values()), title="Word Frequency Distribution", xlabel="Word", ylabel="Count") #, filename="word_frequency_distribution.png")
    common_words = word_freq.most_common(20)
    bar_plot(dict(common_words), title="Word Frequency Distribution", xlabel="Word", ylabel="Count", sort_by_values=True) #, filename="word_frequency_distribution.png")

    # some statistics about the word frequency
    print("N unique words: ", len(word_freq))
    print("words frequency Mean: ", np.mean(list(word_freq.values())))
    print("words frequency Median: ", np.median(list(word_freq.values())))
    print("words frequency Std: ", np.std(list(word_freq.values())))

    # check if there is a correlation between the words and the labels
    for i in range(len(common_words)):
        word_label_corr(transcriptions, labels, common_words[i][0])

    # parameters distribution
    params_freq = get_params_distribution(params, param_classes)
    bar_plot(params_freq, title="Parameter Frequency Distribution", xlabel="Parameter", ylabel="Count", sort_by_values=True) #, filename="param_frequency_distribution.png")
    print("Parameters Distribution: ", params_freq)

    # parameters label distribution
    param_label_freq = get_param_label_distribution(params, labels, param_classes)
    plot_param_label_freq(param_label_freq, param_classes) #, filename="param_label_distribution.png")
    print("Parameters Label Distribution: ", param_label_freq)


if __name__ == '__main__':
    default_data_path = "/Users/omernagar/Documents/Projects/wsc-interview/scripts/data/action_enrichment_ds_home_exercise.csv"
    default_params_file = "/Users/omernagar/Documents/Projects/wsc-interview/scripts/data/params_list.csv"

    # get params from the command line
    parser = argparse.ArgumentParser(description='Perform EDA on the dataset')

    parser.add_argument('--data_path', type=str, default=default_data_path, help='Path to the dataset')
    parser.add_argument('--params_file', type=str, default=default_params_file, help='Path to the parameters file')
    args = parser.parse_args()

    run(args.data_path, args.params_file)