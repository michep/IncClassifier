"""
Classifier SOAP Service
"""
import logging
from wsgiref.simple_server import make_server
from spyne import Application, rpc, ServiceBase, Unicode
from spyne.protocol.soap import Soap11
from spyne.server.wsgi import WsgiApplication
import helper
import config


class EmailIncClassifierService(ServiceBase):
    """ EmailIncClassifierService """

    @rpc(Unicode, _returns=(Unicode, Unicode, Unicode, Unicode), _out_variable_names=config.CLASSCOLS)
    def classify(self, text):
        """ EmailIncClassifierService """
        normtext = helper.normalize_str(text)
        x = VECTORIZER.transform([normtext])
        model = MODEL[config.CLASSIFIER]
        pred = []
        for classcol in config.CLASSCOLS:
            pred.append(model[classcol].predict(x))
        return pred[0][0], pred[1][0], pred[2][0], pred[3][0]


APPLICATION = Application([EmailIncClassifierService], 'org.michep.inclassifier.soap',
                          in_protocol=Soap11(validator='lxml'),
                          out_protocol=Soap11())

WSGI_APPLICATION = WsgiApplication(APPLICATION)


if __name__ == '__main__':
    MODEL = helper.load_pkl("model.pkl")
    VECTORIZER = MODEL["Tfidf"]

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger('spyne.protocol.xml').setLevel(logging.DEBUG)

    logging.info("listening to http://127.0.0.1:8000")
    logging.info("wsdl is at: http://localhost:8000/?wsdl")

    SERVER = make_server('127.0.0.1', 8000, WSGI_APPLICATION)
    SERVER.serve_forever()
