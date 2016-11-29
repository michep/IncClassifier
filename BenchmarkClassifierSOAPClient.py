"""
Benchmark Classifier SOAP Client
"""
import argparse
from time import time
from xml.sax.saxutils import escape
import numpy
import helper
from suds.client import Client
import config


def benchmark_soap_multiple(url, X_test_text, y_test_mul):
    """ benchmark_SOAP_multiple """
    pred = []
    client = Client(url)
    for text in X_test_text:
        pred.append(numpy.array(client.service.classify(escape(text)))[:, 1].tolist())

    res_mul = numpy.array(pred)
    test_mul = numpy.array(y_test_mul)
    score = helper.multiple_score(res_mul, test_mul)
    print("SOAP multiple accuracy:\t{:0.3f}".format(score))


def main():
    """ main """
    apar = argparse.ArgumentParser(description="Benchmark Incident Classifier")
    apar.add_argument("-f", "--file", required=True)
    apar.add_argument("-u", "--url", required=True)
    args = apar.parse_args()
    csv_filename = args.file
    service_url = args.url
    test = helper.load_csv(csv_filename)
    t0 = time()
    test[config.NORMTEXTCOL] = test[config.TEXTCOL].apply(helper.normalize_str)
    print("normalization done:\t{:0.3f}s".format((time() - t0)))
    benchmark_soap_multiple(service_url, test[config.NORMTEXTCOL], test[list(config.CLASSCOLS)])


if __name__ == "__main__":
    main()
