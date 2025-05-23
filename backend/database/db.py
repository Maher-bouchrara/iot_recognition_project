# database/db.py

import pymongo
from config import MONGO_URI, DATABASE_NAME

client = pymongo.MongoClient(MONGO_URI)
db = client[DATABASE_NAME]  # Connexion à la base de données
