# routes/intruder_routes.py

from flask import Blueprint, jsonify
from database.db import db
from config import COLLECTION_NAME

intruder_routes = Blueprint("intruder_routes", __name__)

@intruder_routes.route("/intruders", methods=["GET"])
def get_intruders():
    """Récupérer tous les intrus de la base de données"""
    intruders_collection = db[COLLECTION_NAME]
    intruders = list(intruders_collection.find({}, {"_id": 0}))  # Exclure _id
    return jsonify(intruders)
