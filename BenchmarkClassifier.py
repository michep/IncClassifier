"""
Benchmark Classifier
"""
import argparse
from time import time
from sklearn import metrics
import numpy
import helper


MODEL = {}
TEXTCOL = "Text"
NORMTEXTCOL = "NormText"
CLASSCOLS = ("OperCat", "ProdCat", "Impact", "Type")
VECTORIZER = None
CLASSIFIER = "PassiveAggressive"


def benchmark_single(clf, X_test, y_test):
    """ benchmark_single """
    t0 = time()
    pred = clf.predict(X_test)
    print("test time:\t{:0.3f}s".format(time() - t0))
    metr_score = metrics.accuracy_score(y_test, pred)
    print("single accuracy:\t{:0.3f}".format(metr_score))


def benchmark_multiple(model, X_test, y_test_mul):
    """ benchmark_multiple """
    pred = []
    for classcol in CLASSCOLS:
        pred.append(model[classcol].predict(X_test))

    res_mul = numpy.array(pred).transpose()
    test_mul = numpy.array(y_test_mul)
    score = helper.multiple_score(res_mul, test_mul)
    print("multiple accuracy:\t{:0.3f}".format(score))


def main():
    """ main """
    global MODEL
    apar = argparse.ArgumentParser(description="Benchmark Incident Classifier")
    apar.add_argument("-m", "--model", required=True)
    apar.add_argument("-f", "--file", required=True)
    args = apar.parse_args()
    model_filename = args.model
    csv_filename = args.file
    t0 = time()
    MODEL = helper.load_pkl(model_filename)
    print("model loaded:\t{:0.3f}s".format((time() - t0)))
    test = helper.load_csv(csv_filename)
    t0 = time()
    test[NORMTEXTCOL] = test[TEXTCOL].apply(helper.normalize_str)
    print("normalization done:\t{:0.3f}s".format((time() - t0)))
    vectorizer = MODEL["Tfidf"]
    X_test = vectorizer.transform(test[NORMTEXTCOL])
    benchmark_multiple(MODEL[CLASSIFIER], X_test, test[list(CLASSCOLS)])


if __name__ == "__main__":
    main()
