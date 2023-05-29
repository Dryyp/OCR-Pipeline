from flask import Blueprint

routes = Blueprint('routes', __name__)

@routes.route('/')
def index():
    return "This is purely to test the API endpoint."

from .predict import predict
routes.register_blueprint(predict)