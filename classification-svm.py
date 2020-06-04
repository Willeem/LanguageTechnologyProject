# Language Technology Project, Final Project
# by Martijn Baas, Willem Datema, Stijn Eikelboom and Elvira Slaghekke

from collections import Counter, OrderedDict, defaultdict
from pprint import pprint
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
import numpy as np
import string
import pickle
import os


def load_corpus(phase, cache=True):
    """
    Loads the given corpus file to a document and label list.
    :param phase: String representing the current phase name.
    :param cache: Boolean expressing whether data should be loaded from and saved to cache.
    :return: Two lists, one of documents and one of their labels.
    """
    if phase not in ('train', 'dev', 'test'):
        raise ValueError("Phase should be one of 'train', 'dev', 'test'")

    print('\n=> Loading {} corpus...'.format(phase))

    cache_patt = 'cache/{}.corpus'
    cache_name = cache_patt.format(phase)

    if os.path.isfile(cache_name) and cache:
        print('Using cache...')
        X, Y = pickle.load(open(cache_name, "rb"))

    else:
        print('Reading file...')
        X = []
        Y = []
        paths = {'deepL': 'data/{}_deepL_en.txt',
                 'src': 'data/{}_src_en.txt'}
        for label, path in paths.items():
            with open(path.format(phase)) as corpus:
                for line in corpus:
                    X.append(line.strip())
                    Y.append(label)

        if cache:
            pickle.dump((X, Y), open(cache_name, "wb"))

    return X, Y


def data_description(X, Y, phase):
    """
    Prints an overview of the proportion and the distribution of the corpus.
    :param X: List of document features to consider.
    :param Y: List of document labels to consider.
    """
    if phase not in ('train', 'dev', 'test'):
        raise ValueError("Phase should be one of 'train', 'dev', 'test'")

    # Determine proportions and distributions
    total_docs = len(X)
    label_counts = Counter()
    for label in Y:
        label_counts.update([label])

    # Print the results
    print('\n== Data Description ({}) =='.format(phase))
    print('-- Overall proportion --')
    print('{:<18}{:<10}'.format('Total articles:', total_docs))

    print('\n-- Labels --')
    for label, count in sorted(label_counts.items()):
        print('{:<18}{:<10}{:<10.3f}'.format('{}:'.format(label), count, count / total_docs))


def preprocess(X, phase, stopwords='english', lowercase=True, rm_punctuation=True, cache=True):
    """
    Filters and transforms dictionaries of features based on the given parameters.
    :param X: List of strings representing documents.
    :param phase: String representing the current phase name.
    :param stopwords: String referring to the list of stopwords from NLTK to use.
    :param lowercase: Boolean expressing whether considered text should be lowercased.
    :param rm_punctuation: Boolean expressing whether punctuation should be removed from considered text.
    :param cache: Boolean expressing whether data should be loaded from and saved to cache.
    :return: List of strings representing filtered documents.
    """
    if phase not in ('train', 'dev', 'test'):
        raise ValueError("Phase should be one of 'train', 'dev', 'test'")

    print('\n=> Preprocessing {} corpus...'.format(phase))

    cache_patt = 'cache/{}-{}-{}-{}.processed'
    cache_name = cache_patt.format(phase, stopwords, lowercase, rm_punctuation)

    if os.path.isfile(cache_name) and cache:
        print('Using cache...')
        Xfiltered = pickle.load(open(cache_name, "rb"))

    else:
        # Initialize removal set
        remove = set()

        # Add stopwords to removal set
        if stopwords:
            stopset = set(nltk_stopwords.words(stopwords))
            remove.update(stopset)

        # Add punctuation to removal set
        if rm_punctuation:
            punctuation = set(string.punctuation)
            remove.update(punctuation)

        # Filter tokens
        Xfiltered = []
        total_docs = len(X)
        for doc_num, doc in enumerate(X, 1):
            progress = round(doc_num / total_docs * 100, 1)
            print('\rDocument {}/{} ({}%)...'.format(doc_num, total_docs, progress), end='')

            # Lowercase
            if lowercase:
                doc = doc.lower()

            # Tokenize
            doc = word_tokenize(doc)

            # Perform removal
            doc = [w for w in doc if w.lower() not in remove]

            Xfiltered.append(' '.join(doc))

        print()

        if cache:
            pickle.dump(Xfiltered, open(cache_name, "wb"))

    return Xfiltered


def get_vectorizer(tfidf=False, **v_settings):
    """
    Creates an instance of either TfidfVectorizer or CountVectorizer.
    :param tfidf: Whether to use TfidfVectorizer or CountVectorizer.
    :param v_settings: Parameters to pass on to the vectorizer.
    :return: Instance of chosen vectorizer.
    """
    if tfidf:
        return TfidfVectorizer(**v_settings)
    else:
        return CountVectorizer(**v_settings)


def get_pipeline(feat, vec, tfidf=False):
    """
    Creates a Pipeline of the given feature transformer, vectorizer and possibly TfIdfTransformer.
    :param feat: Instance of custom feature transformer to include.
    :param vec: Instance of vectorizer to include.
    :param tfidf: Boolean expressing whether the TfIdfTransformer should be included.
    :return: Pipeline of the given feature transformer, vectorizer and possibly TfIdfTransformer.
    """
    pipeline = [('feat', feat),
                ('vec', vec)]

    if tfidf:
        pipeline.append(('tfidf', TfidfTransformer()))

    return Pipeline(pipeline)


def get_classifier(classifier, m_settings):
    """
    Easily provides instance of a selected classifier from SciKit Learn.
    :param classifier: String referring to classifier, one of
    LinearSVC, LogisticRegression, RandomForestClassifier or DummyClassifier.
    :param m_settings: Dictionary of settings to pass on as arguments to the classifier.
    :return: Instance of the chosen classifier.
    """
    if classifier == 'LinearSVC':
        clf = LinearSVC(**m_settings)
    elif classifier == 'LogisticRegression':
        clf = LogisticRegression(**m_settings)
    elif classifier == 'RandomForestClassifier':
        clf = RandomForestClassifier(**m_settings)
    elif classifier == 'DummyClassifier':
        clf = DummyClassifier(random_state=48, **m_settings)
    else:
        raise ValueError("Classifier should be one of 'LinearSVC', 'LogisticRegression',"
                         "'RandomForestClassifier' or 'DummyClassifier'")

    return clf


def get_model(vec, clf):
    """
    Creates a Pipeline of the given vectorizer and classifier.
    :param vec: Instance of vectorizer to include.
    :param clf: Instance of classifier to include.
    :return: Pipeline of the given vectorizer and classifier.
    """
    return Pipeline([('vec', vec),
                     ('clf', clf)])


def ml_regular(model, Xtrain=None, Ytrain=None, Xtest=None, Ytest=None):
    """
    Perform single Machine Learning run on given data.
    Print Classification Report and Confusion Matrix afterwards.
    :param model: Instance of Pipeline containing vectorizer and classifier to be used.
    :param Xtrain: List of features to be used in training the model.
    :param Ytrain: List of correct labels for the training features.
    :param Xtest: List of features to be used in testing the model.
    :param Ytest: List of correct labels for the testing features.
    """
    if not (Xtrain and Ytrain) and not (Xtest and Ytest):
        raise ValueError('X and Y must be specified at least for training or testing')

    # Train the model
    if Xtrain and Ytrain:
        model.fit(Xtrain, Ytrain)

    # Predict Xtest
    if Xtest and Ytest:
        Ypred = model.predict(Xtest)

        # Perform evaluation
        print('\n== Regular Machine Learning ==')
        print('-- Classification Report --')
        print(classification_report(Ytest, Ypred, model.classes_))
        print('-- Confusion Matrix --')
        cf_matrix(Ytest, Ypred, model.classes_)

        return Ypred


def ml_crossval(model, X, Y, cv=10):
    """
    Perform Machine Learning on given data using K-fold Cross Validation.
    Print Classification Report and Confusion Matrix afterwards.
    :param model: Instance of Pipeline containing vectorizer and classifier to be used.
    :param X: List of features to be considered in Cross Validation.
    :param Y: List of correct labels for the given features.
    :param cv: Number of folds to consider in Cross Validation.
    """
    X = np.array(X)
    Y = np.array(Y)
    kf = KFold(n_splits=cv)
    Ytest_all = []
    Ypred_all = []

    for train_index, test_index in kf.split(X):
        # Set Xtrain, Xtest, Ytrain and Ytest
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]

        # Train the model
        model.fit(Xtrain, Ytrain)

        # Predict for Xtest
        Ypred = model.predict(Xtest)

        # Save for evaluation
        Ytest_all += list(Ytest)
        Ypred_all += list(Ypred)

    # Perform overall evaluation
    print('\n== Cross Validation ==')
    print('-- Classification Report --')
    print(classification_report(Ytest_all, Ypred_all, labels=model.classes_))
    print('-- Confusion Matrix --')
    cf_matrix(Ytest_all, Ypred_all, model.classes_)

    return Ypred_all


def cf_matrix(Ytest, Ypred, labels):
    """
    Generate and print confusion matrix based on given labels.
    :param Ytest: List of correct labels for the dataset.
    :param Ypred: List of predicted labels for the dataset.
    :param labels: Iterable of labels used to classify the dataset.
    """
    cf = confusion_matrix(Ytest, Ypred, labels=labels)
    cf_new = cf.tolist()

    max_label = max(labels, key=len)
    max_chars = str(len(max_label) + 1)
    format_header = "{:>" + max_chars + "}"
    format_str = "{:" + max_chars + "}"
    for i in range(len(labels)):
        format_header += "{:>" + max_chars + "}"
        format_str += "{:" + max_chars + "}"

    for i in range(len(cf_new)):
        cf_new[i].insert(0, labels[i])
    print(format_header.format("", *labels))
    for row in cf_new:
        print(format_str.format(*row))
    print()


def most_informative_features(vec, clf, labels, n=10):
    """
    Extract and print the Most Informative Features considered by the given model.
    :param vec: Vectorizer instance that was executed on the dataset.
    :param clf: Classifier that was trained on the dataset.
    :param labels: Iterable of labels used to classify the dataset.
    :param n: Number of features to print for each label.
    """
    print('= Most Informative Features =')
    coef = clf.coef_
    feature_names = vec.get_feature_names()
    for i in range(0, len(coef)):
        coefs_with_fns = sorted(zip(coef[i], feature_names))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        if len(coef) > 2:
            print(labels[i])
        else:
            print(labels[1])
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print("\t%.4f\t%-25s\t\t%.4f\t%-25s" % (coef_1, fn_1, coef_2, fn_2))


def classify(p_settings, c_settings, v_settings, m_settings):
    """
    Perform all required steps for classification of sentiment labels in code-mixed text.
    :param p_settings: Dictionary of parameters for the preprocessing function.
    :param c_settings: Dictionary of parameters to tweak classification steps to execute.
    :param v_settings: Dictionary of parameters for the vectorizer.
    :param m_settings: Dictionary of parameters for the classifier.
    """
    # Print out settings
    print('== Settings ==')
    print('-- Feature Generation --')
    for setting, value in p_settings.items():
        print('{:30} {}'.format(setting, value))

    print('\n-- Classifier ({}) --'.format(c_settings['classifier']))
    if not m_settings:
        print('Default settings')
    for setting, value in m_settings.items():
        print('{:30} {}'.format(setting, value))

    # Get train corpus
    Xtrain, Ytrain, = load_corpus('train')
    data_description(Xtrain, Ytrain, 'train')

    # Get dev corpus
    Xdev, Ydev = load_corpus('dev')
    data_description(Xdev, Ydev, 'dev')

    # Perform preprocessing
    Xtrain = preprocess(Xtrain, 'train', **p_settings)
    Xdev = preprocess(Xdev, 'dev', **p_settings)

    # Merge train and dev data
    Xtrain_dev = Xtrain + Xdev
    Ytrain_dev = Ytrain + Ydev

    # Perform training and evaluation
    print('\n=> Fitting model...')

    # Select vectorizer
    vec = get_vectorizer(True, **v_settings)

    # Select classifier
    clf = get_classifier(c_settings['classifier'], m_settings)

    # Create model
    model = get_model(vec, clf)

    # Fit and predict
    if c_settings['ml_regular']:
        ml_regular(model, Xtrain, Ytrain, Xdev, Ydev)
    if c_settings['ml_crossval']:
        ml_crossval(model, Xtrain_dev, Ytrain_dev)
    if c_settings['final_test']:
        print('== FINAL TEST ==')
        print('=> Fitting for final test...')
        ml_regular(model, Xtrain=Xtrain, Ytrain=Ytrain)

        # Load and preprocess testing data
        Xtest, Ytest = load_corpus('test')
        Xtest = preprocess(Xtest, 'test', **p_settings)

        # Perform evaluation
        ml_regular(model, Xtest=Xtest, Ytest=Ytest)

    # Print most informative features
    if c_settings['informative_features'] and c_settings['classifier'] != 'DummyClassifier':
        most_informative_features(vec, clf, model.classes_, n=c_settings['informative_features'])


def main():
    # Settings
    p_settings = {
        'lowercase': False,
        'stopwords': False,
        'rm_punctuation': False
    }

    c_settings = {
        'classifier': 'LinearSVC',
        'ml_regular': True,
        'ml_crossval': False,
        'final_test': False,
        'informative_features': 25
    }

    v_settings = {
        'ngram_range': (1, 3)
    }

    m_settings = {}

    # Perform classification
    classify(p_settings, c_settings, v_settings, m_settings)


if __name__ == "__main__":
    main()
