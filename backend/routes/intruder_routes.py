# routes/intruder_routes.py

import os
import shutil
from flask import Blueprint, jsonify, request
from database.db import db
from config import COLLECTION_NAME

intruder_routes = Blueprint("intruder_routes", __name__)

@intruder_routes.route("/intruders", methods=["GET"])
def get_intruders():
    """Récupérer tous les intrus de la base de données"""
    try:
        intruders_collection = db[COLLECTION_NAME]
        intruders = list(intruders_collection.find({}, {"_id": 0}))  # Exclure _id
        return jsonify({
            "success": True,
            "data": intruders,
            "count": len(intruders)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@intruder_routes.route("/intruders/grouped", methods=["GET"])
def get_intruders_grouped():
    """Récupérer tous les intrus regroupés par intruder_id"""
    try:
        intruders_collection = db[COLLECTION_NAME]
        
        # Pipeline d'agrégation pour regrouper par intruder_id
        pipeline = [
            {
                "$match": {
                    "classified": True,  # Seulement les intrus classifiés
                    "intruder_id": {"$exists": True}
                }
            },
            {
                "$group": {
                    "_id": "$intruder_id",
                    "intruder_id": {"$first": "$intruder_id"},
                    "images": {
                        "$push": {
                            "image_number": "$image_number",
                            "timestamp": "$timestamp",
                            "image_path": "$image_path",
                            "image_base64": "$image_base64"
                        }
                    },
                    "total_images": {"$sum": 1},
                    "first_detection": {"$min": "$timestamp"},
                    "last_detection": {"$max": "$timestamp"}
                }
            },
            {
                "$sort": {"intruder_id": 1}
            }
        ]
        
        grouped_intruders = list(intruders_collection.aggregate(pipeline))
        
        return jsonify({
            "success": True,
            "data": grouped_intruders,
            "total_intruders": len(grouped_intruders),
            "total_images": sum(intruder["total_images"] for intruder in grouped_intruders)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@intruder_routes.route("/intruders/<intruder_id>", methods=["GET"])
def get_intruder_by_id(intruder_id):
    """Récupérer un intrus spécifique par son ID"""
    try:
        intruders_collection = db[COLLECTION_NAME]
        
        intruder_images = list(intruders_collection.find(
            {"intruder_id": intruder_id, "classified": True},
            {"_id": 0}
        ).sort("image_number", 1))
        
        if not intruder_images:
            return jsonify({
                "success": False,
                "error": f"Intrus avec l'ID '{intruder_id}' non trouvé"
            }), 404
        
        return jsonify({
            "success": True,
            "intruder_id": intruder_id,
            "data": intruder_images,
            "total_images": len(intruder_images),
            "first_detection": intruder_images[0]["timestamp"] if intruder_images else None,
            "last_detection": intruder_images[-1]["timestamp"] if intruder_images else None
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@intruder_routes.route("/intruders/<intruder_id>", methods=["DELETE"])
def delete_intruder(intruder_id):
    """Supprimer un intrus et son dossier physique"""
    try:
        intruders_collection = db[COLLECTION_NAME]
        
        # Vérifier si l'intrus existe
        intruder_exists = intruders_collection.find_one({"intruder_id": intruder_id})
        if not intruder_exists:
            return jsonify({
                "success": False,
                "error": f"Intrus avec l'ID '{intruder_id}' non trouvé"
            }), 404
        
        # Récupérer le chemin du dossier depuis la base de données
        sample_document = intruders_collection.find_one({"intruder_id": intruder_id})
        if sample_document and "image_path" in sample_document:
            # Extraire le chemin du dossier depuis le chemin de l'image
            image_path = sample_document["image_path"]
            # Exemple: "Intrus\Classes\Intrus_009\Intrus_009_001_20250523_234355.jpg"
            # On veut: "Intrus\Classes\Intrus_009"
            folder_path = os.path.dirname(image_path)
            
            # Supprimer le dossier physique s'il existe
            if os.path.exists(folder_path):
                try:
                    shutil.rmtree(folder_path)
                    folder_deleted = True
                    folder_message = f"Dossier '{folder_path}' supprimé avec succès"
                except Exception as folder_error:
                    folder_deleted = False
                    folder_message = f"Erreur lors de la suppression du dossier: {str(folder_error)}"
            else:
                folder_deleted = False
                folder_message = f"Dossier '{folder_path}' non trouvé sur le disque"
        else:
            folder_deleted = False
            folder_message = "Impossible de déterminer le chemin du dossier"
        
        # Supprimer tous les documents de cet intrus de la base de données
        delete_result = intruders_collection.delete_many({"intruder_id": intruder_id})
        
        return jsonify({
            "success": True,
            "message": f"Intrus '{intruder_id}' supprimé avec succès",
            "deleted_from_db": delete_result.deleted_count,
            "folder_deleted": folder_deleted,
            "folder_message": folder_message
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@intruder_routes.route("/intruders/cleanup-temp", methods=["DELETE"])
def cleanup_temp_intruders():
    """Nettoyer les captures temporaires (non classifiées)"""
    try:
        intruders_collection = db[COLLECTION_NAME]
        
        # Supprimer les documents temporaires
        delete_result = intruders_collection.delete_many({"classified": False})
        
        return jsonify({
            "success": True,
            "message": "Captures temporaires nettoyées",
            "deleted_count": delete_result.deleted_count
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500