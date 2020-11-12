from flask import Flask
from flask_cors import CORS

HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello():
	return "Hello world"



if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)
