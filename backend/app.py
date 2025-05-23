# app.py

from flask import Flask
from routes.intruder_routes import intruder_routes

app = Flask(__name__)

# Enregistrement des routes
app.register_blueprint(intruder_routes)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible sur le réseau local
