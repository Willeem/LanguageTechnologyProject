# Language Technology Project, Final Project
# by Martijn Baas, Willem Datema, Stijn Eikelboom and Elvira Slaghekke


import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from simpletransformers.classification import ClassificationModel
from os import path
import argparse


def read_data():
    train_en = [[line.rstrip(), 0] for line in open('data/train_src_en.txt', encoding='utf-8').readlines()]
    train_deepl = [[line.rstrip(), 1] for line in open('data/train_deepL_en.txt', encoding='utf-8').readlines()]
    dev_en = [[line.rstrip(), 0] for line in open('data/dev_src_en.txt', encoding='utf-8').readlines()]
    dev_deepl = [[line.rstrip(), 1] for line in open('data/dev_deepL_en.txt', encoding='utf-8').readlines()]
    test_en = [[line.rstrip(), 0] for line in open('data/test_src_en.txt', encoding='utf-8').readlines()]
    test_deepl = [[line.rstrip(), 1] for line in open('data/test_deepL_en.txt', encoding='utf-8').readlines()]

    return train_en + train_deepl, dev_en + dev_deepl, test_en + test_deepl


def parse_args():
    parser = argparse.ArgumentParser(description='Give arguments for running the classifier.')
    parser.add_argument('--num_epochs', type=int, help='Specify amount of epochs')
    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(212)
    arguments = parse_args()
    num_epochs = arguments.num_epochs
    train, dev, test = read_data()

    # Turn into pandas dataframes so SimpleTransformers can use the data
    train = pd.DataFrame(train)
    dev = pd.DataFrame(dev)
    test = pd.DataFrame(test)

    if path.exists('storage/xlnet_outputs_{}_epoch/'.format(num_epochs)):
        model_XLNet = ClassificationModel('bert', 'storage/xlnet_outputs_{}_epoch/'.format(num_epochs))
        eval_df = test

    else:
        args = {
            'num_train_epochs': num_epochs,
            'train_batch_size': 32,
            'eval_batch_size': 32,
            'output_dir': 'storage/xlnet_outputs_{}_epoch/'.format(num_epochs),
            'cache_dir': 'storage/cache_xlnet_{}/'.format(num_epochs)
        }

        # Train the model
        model_XLNet = ClassificationModel('xlnet', 'xlnet-base-cased', use_cuda=True, args=args)
        model_XLNet.train_model(train)
        eval_df = dev

    # Evaluate the model
    result, model_XLNet_outputs, wrong_predictions_XLNet = model_XLNet.eval_model(eval_df, cr=classification_report,
                                                                                  cm=confusion_matrix,
                                                                                  acc=accuracy_score,
                                                                                  verbose=True)

    print(result['cr'])  # Classification Report
    print(result['cm'])  # Confusion Matrix
    print(result['acc'])  # Accuracy Score
