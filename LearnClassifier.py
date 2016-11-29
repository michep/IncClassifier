"""
Learn Classifier
"""
import argparse
import copy
import multiprocessing
from typing import List
from time import time
import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import helper
import config

PARTITIONLEN = 100
NUMCORES = 3
VERBOSE = None
CLASSIFIERS = (("PassiveAggressive", PassiveAggressiveClassifier(n_jobs=-1)),
               ("SGD", SGDClassifier(n_jobs=-1)),
               ("RandomForest", RandomForestClassifier(n_jobs=-1)))


def print_verbose(line: str) -> None:
    """ pring_verbose """
    if VERBOSE:
        print(line)


def normalize_data(data: List[numpy.ndarray]) -> List[numpy.ndarray]:
    """ normalize_data """
    data[config.NORMTEXTCOL] = data[config.TEXTCOL].apply(helper.normalize_str)
    return data


def main():
    """ main """
    global VERBOSE
    VECTORIZER = TfidfVectorizer()
    apar = argparse.ArgumentParser(description="Learn Incident Classifier")
    apar.add_argument("-f", "--file", required=True)
    apar.add_argument("-o", "--out")
    apar.add_argument("-v", "--verbose", action="store_true")
    args = apar.parse_args()
    VERBOSE = args.verbose
    csv_filename = args.file
    pkl_filename = args.out if args.out else "model.pkl"
    data = helper.load_csv(csv_filename)
    t0 = time()

    # data[NORMTEXTCOL] = data[TEXTCOL].apply(helper.normalize_str)

    numpartitions = (len(data) // PARTITIONLEN) + 1
    split_data = numpy.array_split(data, numpartitions)
    pool = multiprocessing.Pool(NUMCORES)
    data = pandas.concat(pool.map(normalize_data, split_data))
    pool.close()
    pool.join()

    print_verbose("normalization done:\t{:0.3f}s".format((time() - t0)))
    t0 = time()
    X_learn = VECTORIZER.fit_transform(data[config.NORMTEXTCOL])
    print_verbose("fit done\t{:0.3f}s".format((time() - t0)))
    config.MODEL["Tfidf"] = VECTORIZER
    for classifier_name, classifier in CLASSIFIERS:
        config.MODEL[classifier_name] = {}
        print("="*40)
        print(classifier_name)
        print(classifier)
        for classcol in config.CLASSCOLS:
            print("training on\t{:s}".format(classcol))
            t0 = time()
            y_learn = numpy.array(data[classcol])
            classifier.fit(X_learn, y_learn)
            print_verbose("training done\t{:0.3f}s".format((time() - t0)))
            config.MODEL[classifier_name][classcol] = copy.copy(classifier)
    helper.save_pkl(config.MODEL, pkl_filename)

if __name__ == "__main__":
    main()
