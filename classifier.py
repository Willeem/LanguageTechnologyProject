# filename: classifier.py
#
# Authors: Martijn Baas, Willem Datema, Stijn Eikelboom and Elvira Slaghekke


import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel


def read_data():
    train_en = [[line.rstrip(), 0] for line in open('data/train_src_en.txt', encoding='utf-8').readlines()]
    train_deepl = [[line.rstrip(), 1] for line in open('data/train_deepL_en.txt', encoding='utf-8').readlines()]
    dev_en = [[line.rstrip(), 1] for line in open('data/dev_src_en.txt', encoding='utf-8').readlines()]
    dev_deepl = [[line.rstrip(), 1] for line in open('data/dev_deepL_en.txt', encoding='utf-8').readlines()]
    test_en = [[line.rstrip(), 0] for line in open('data/test_src_en.txt', encoding='utf-8').readlines()]
    test_deepl = [[line.rstrip(), 1] for line in open('data/test_deepl_en.txt', encoding='utf-8').readlines()]

    return train_en + train_deepl, dev_en + dev_deepl, test_en + test_deepl


if __name__ == '__main__':
    # TODO: voeg argparse toe wat de input beter kan verwerken
    np.random.seed(212)

    train, dev, test = read_data()

    # shuffle data
    np.random.shuffle(train)
    np.random.shuffle(dev)
    np.random.shuffle(test)

    # Turn into pandas dataframes so SimpleTransformers can use the data
    train = pd.DataFrame(train)
    dev = pd.DataFrame(dev)
    test = pd.DataFrame(test)

    # TODO: Hier kunnen we wat data-info toevoegen (label distribution e.d.)

    model = ClassificationModel('roberta', 'roberta-base')

    # Train the model
    model.train_model(train)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(dev)


