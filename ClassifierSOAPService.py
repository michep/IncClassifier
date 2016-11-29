import logging
from wsgiref.simple_server import make_server
from spyne import Application, rpc, ServiceBase, Unicode, AnyDict
from spyne.protocol.soap import Soap11
from spyne.server.wsgi import WsgiApplication
import helper

MODEL = {}
TEXTCOL = "Text"
NORMTEXTCOL = "NormText"
CLASSCOLS = ("OperCat", "ProdCat", "Impact", "Type")
VECTORIZER = None
CLASSIFIER = "PassiveAggressive"


class EmailIncClassifierService(ServiceBase):
    @rpc(Unicode, _returns=(Unicode, Unicode, Unicode, Unicode), _out_variable_names=CLASSCOLS)
    def classify(ctx, text):
        """ EmailIncClassifierService """
        global VECTORIZER, MODEL
        normtext = helper.normalize_str(text)
        x = VECTORIZER.transform([normtext])
        model = MODEL[CLASSIFIER]
        pred = []
        for classcol in CLASSCOLS:
            pred.append(model[classcol].predict(x))
        return pred[0][0], pred[1][0], pred[2][0], pred[3][0]


application = Application([EmailIncClassifierService], 'org.michep.inclassifier.soap',
                          in_protocol=Soap11(validator='lxml'),
                          out_protocol=Soap11())

wsgi_application = WsgiApplication(application)


if __name__ == '__main__':

    MODEL = helper.load_pkl("model.pkl")
    VECTORIZER = MODEL["Tfidf"]

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger('spyne.protocol.xml').setLevel(logging.DEBUG)

    logging.info("listening to http://127.0.0.1:8000")
    logging.info("wsdl is at: http://localhost:8000/?wsdl")

    server = make_server('127.0.0.1', 8000, wsgi_application)
    server.serve_forever()