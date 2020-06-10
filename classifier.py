# filename: classifier.py
#
# Authors: Martijn Baas, Willem Datema, Stijn Eikelboom and Elvira Slaghekke


import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from simpletransformers.classification import ClassificationModel


def read_data():
    train_en = [[line.rstrip(), 0] for line in open('data/train_src_en.txt', encoding='utf-8').readlines()]
    train_deepl = [[line.rstrip(), 1] for line in open('data/train_deepL_en.txt', encoding='utf-8').readlines()]
    dev_en = [[line.rstrip(), 0] for line in open('data/dev_src_en.txt', encoding='utf-8').readlines()]
    dev_deepl = [[line.rstrip(), 1] for line in open('data/dev_deepL_en.txt', encoding='utf-8').readlines()]
    test_en = [[line.rstrip(), 0] for line in open('data/test_src_en.txt', encoding='utf-8').readlines()]
    test_deepl = [[line.rstrip(), 1] for line in open('data/test_deepL_en.txt', encoding='utf-8').readlines()]

    return train_en + train_deepl, dev_en + dev_deepl, test_en + test_deepl


if __name__ == '__main__':
    # TODO: voeg argparse toe wat de input beter kan verwerken
    np.random.seed(212)

    train, dev, test = read_data()

    # Turn into pandas dataframes so SimpleTransformers can use the data
    train = pd.DataFrame(train)
    dev = pd.DataFrame(dev)
    test = pd.DataFrame(test)

    # TODO: Hier kunnen we wat data-info toevoegen (label distribution e.d.)

    model_BERT = ClassificationModel('bert', 'bert-base-cased' , use_cuda=True,  args={'num_train_epochs':10, 'train_batch_size': 32, 'eval_batch_size':32, 'output_dir': 'bert_outputs_10_epoch/', 'cache_dir': 'cache_bert_10/'})
    # Train the model
    model_BERT.train_model(train)

    # Evaluate the model
    result, model_BERT_outputs, wrong_predictions_BERT = model_BERT.eval_model(dev, cr=classification_report, cm=confusion_matrix, acc=accuracy_score, verbose=True)
    print(result['cr'])  # Classification Report
    print(result['cm'])  # Confusion Matrix
    print(result['acc'])  # Accuracy Score  


