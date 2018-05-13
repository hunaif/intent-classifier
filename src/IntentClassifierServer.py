from klein import Klein
from IntentClassifier import *
from Constants import PORT,URL
class IntentClassifierServer:
    app = Klein()
    @app.route("/predict", methods=['POST'])
    def get_prediction(request):
        request.setHeader('Content-Type', 'application/json')
        request_params = json.loads(
                request.content.read().decode('utf-8', 'strict'))

        if 'query' in request_params:
            query = request_params.pop('query')
        else:
            return(json.dumps({"bad request":"query field is manadatory in request"}))

        prediction = IntentClassifier.predict(query)
        return json.dumps(prediction)

if __name__ == "__main__":
    IntentClassifierServer.app.run(URL, PORT)