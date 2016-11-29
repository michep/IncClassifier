import argparse
from time import time
from xml.sax.saxutils import escape
from sklearn import metrics
import numpy
import helper
from suds.client import Client


MODEL = {}
TEXTCOL = "Text"
NORMTEXTCOL = "NormText"
CLASSCOLS = ("OperCat", "ProdCat", "Impact", "Type")
VECTORIZER = None
CLASSIFIER = "PassiveAggressive"

def multiple_score(pred_mul, test_mul):
    correct = 0
    for rowi, row in enumerate(test_mul):
        correct_row = 0
        for eli, el in enumerate(row):
            if el == pred_mul[rowi][eli]:
                correct_row += 1
        if correct_row == len(row):
            correct += 1
    score = correct / len(test_mul)
    return score


def benchmark_soap_multiple(url, X_test_text, y_test_mul):
    """ benchmark_SOAP_multiple """
    pred = []
    client = Client(url)
    for text in X_test_text:
        pred.append(numpy.array(client.service.classify(escape(text)))[:, 1].tolist())

    res_mul = numpy.array(pred)
    test_mul = numpy.array(y_test_mul)
    score = multiple_score(res_mul, test_mul)
    print("SOAP multiple accuracy:\t{:0.3f}".format(score))


def main():
    """ main """
    global MODEL
    apar = argparse.ArgumentParser(description="Benchmark Incident Classifier")
    apar.add_argument("-f", "--file", required=True)
    apar.add_argument("-u", "--url", required=True)
    args = apar.parse_args()
    csv_filename = args.file
    service_url = args.url
    test = helper.load_csv(csv_filename)
    t0 = time()
    test[NORMTEXTCOL] = test[TEXTCOL].apply(helper.normalize_str)
    print("normalization done:\t{:0.3f}s".format((time() - t0)))
    benchmark_soap_multiple(service_url, test[NORMTEXTCOL], test[list(CLASSCOLS)])


if __name__ == "__main__":
    main()
