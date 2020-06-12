# Language Technology Project, Final Project
# by Martijn Baas, Willem Datema, Stijn Eikelboom and Elvira Slaghekke

import pandas as pd
import numpy as np
import argparse
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from simpletransformers.classification import ClassificationModel
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for running the classifier.')
    parser.add_argument('--arch', type=str, choices=['bert', 'roberta', 'xlnet'],
                        help='Transformer architecture to use')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--test', action='store_true', help='Run test using trained model')
    return parser.parse_args()


def load_corpus(phase):
    """
    Loads the given corpus file to a document and label list.
    :param phase: String representing the phase for which date should be loaded.
    :return: List of document and label pairs.
    """
    if phase not in ('train', 'dev', 'test'):
        raise ValueError("Phase should be one of 'train', 'dev', 'test'")

    print('=> Loading {} corpus...'.format(phase))

    corpus_data = []
    paths = {0: 'data/{}_src_en.txt',
             1: 'data/{}_deepL_en.txt'}

    for label, path in paths.items():
        with open(path.format(phase), encoding='utf-8') as corpus:
            for line in corpus:
                corpus_data.append([line.rstrip(), label])

    df = pd.DataFrame(corpus_data)
    df.columns = ['text', 'labels']
    return df


def data_description(corpus):
    """
    Prints an overview of the proportion and the distribution of the corpus.
    :param corpus:
    """
    # Split documents and labels
    X, Y = zip(*corpus.values.tolist())

    # Determine proportions and distributions
    total_docs = len(X)
    label_counts = Counter()
    for label in Y:
        label_counts.update([label])

    # Print the results
    print('\n== Data Description ==')
    print('-- Overall proportion --')
    print('{:<7}{:<10}'.format('Total articles:', total_docs))

    print('\n-- Labels --')
    for label, count in sorted(label_counts.items()):
        print('{:<7}{:<10}{:<10.3f}'.format('{}:'.format(label), count, count / total_docs))
    print()


def get_transformer_model(arch):
    """
    Get the name of the pretrained model for the given architecture.
    :param arch: Architecture to return pretrained model for.
    :return: Name of the pretrained model.
    """
    models = {
        'bert': 'bert-base-uncased',
        'roberta': 'roberta-base',
        'xlnet': 'xlnet-base-cased'
    }
    return models[arch]


def train_model(args, output_dir, cache_dir):
    """
    Train a SimpleTransformers model based on the given arguments, save and return it.
    :param args: Arguments as processed by parse_args() containing architecture and epochs.
    :param output_dir: Path to the directory in which the model should be stored.
    :param cache_dir: Path to the directory in which the cache should be stored.
    :return: SimpleTransformers model trained based on the given arguments.
    """
    print('=> Training model...')

    # Set model arguments
    model_args = {
        'num_train_epochs': args.num_epochs,
        'train_batch_size': 32,
        'eval_batch_size': 32,
        'output_dir': output_dir,
        'cache_dir': cache_dir
    }

    # Train the model
    pretrained = get_transformer_model(args.arch)
    model = ClassificationModel(args.arch, pretrained, use_cuda=True, args=model_args)
    train = load_corpus('train')
    model.train_model(train)

    return model


def load_model(args, output_dir):
    """
    Load a trained model from the given output directory.
    :param args: Arguments as processed by parse_args() containing architecture and epochs.
    :param output_dir: Path to the directory in which the trained model is stored.
    :return: SimpleTransformers model loaded from the given output directory.
    """
    print('=> Loading trained model...')

    # Check trained model exists
    if not os.path.exists(output_dir):
        raise FileNotFoundError('Trained model directory not found. Run training first.')

    # Load the model
    model = ClassificationModel(args.arch, output_dir)

    return model


def cf_matrix(cm, labels):
    """
    Generate and print confusion matrix based on given labels.
    :param cm: Confusion matrix as produced by SKLearn confusion_matrix().
    :param labels: Iterable of labels used to classify the dataset.
    """
    cm_new = cm.tolist()
    labels = [str(lab) for lab in labels]
    max_label = len(max(labels, key=len))
    max_number = len(max([str(col) for row in cm for col in row], key=len))
    max_chars = str(max(max_label, max_number) + 1)
    format_header = "{:>" + max_chars + "}"
    format_str = "{:" + max_chars + "}"
    for i in range(len(labels)):
        format_header += "{:>" + max_chars + "}"
        format_str += "{:" + max_chars + "}"
    for i in range(len(cm_new)):
        cm_new[i].insert(0, labels[i])
    print(format_header.format("", *labels))
    for row in cm_new:
        print(format_str.format(*row))
    print()


def evaluate_model(model, phase):
    """
    Evaluate the given model on the given dataframe.
    :param model: SimpleTransformers model to evaluate.
    :param phase: String representing the phase for which date should be loaded.
    """
    # Load corpus
    eval_df = load_corpus(phase)

    # Print data description
    data_description(eval_df)

    print('=> Evaluating model on {} corpus...'.format(phase))
    result, _, _ = model.eval_model(eval_df,
                                    cr=classification_report,
                                    cm=confusion_matrix,
                                    acc=accuracy_score,
                                    verbose=True)

    print('\n-- Classification Report --')
    print(result['cr'])
    print('-- Confusion Matrix --')
    cf_matrix(result['cm'], [0, 1])
    print('-- Accuracy Score --')
    print(result['acc'], end='\n\n')


def main():
    # Set random seed
    np.random.seed(212)

    # Get arguments
    args = parse_args()

    # Set directories
    output_dir = 'storage/{}_outputs_{}_epoch/'.format(args.arch, args.num_epochs)
    cache_dir = 'storage/cache_{}_{}/'.format(args.arch, args.num_epochs)

    if args.test:
        # Test the model
        model = load_model(args, output_dir)

        # Set phase for corpus
        phase = 'test'

    else:
        # Train the model
        model = train_model(args, output_dir, cache_dir)

        # Set phase for corpus
        phase = 'dev'

    # Evaluate the model
    evaluate_model(model, phase)


if __name__ == '__main__':
    main()
