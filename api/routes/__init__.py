from flask import Flask
from flask_cors import CORS

def app():
    app = Flask(__name__)
    CORS(app)
    
    from .predict import routes
    app.register_blueprint(routes)
    
    return app

