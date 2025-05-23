import os
import cv2
import numpy as np
import torch
import pymongo
import time
import base64
import json
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.cluster import DBSCAN

# Connexion à MongoDB
client = pymongo.MongoClient("mongodb+srv://maherbouchrara:maherbouchrara@iotproject.6zy5h.mongodb.net/?retryWrites=true&w=majority&appName=IotProject")
db = client["IotProject"]
collection = db["Intruders"]

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

class KnownFacesManager:
    def __init__(self, images_folder="Images1", cache_file="known_faces_cache.pkl"):
        self.images_folder = images_folder
        self.cache_file = cache_file
        self.known_face_encodings = []
        self.known_face_names = []
        
    def need_to_reload(self):
        """Vérifie si le cache doit être rechargé"""
        if not os.path.exists(self.cache_file):
            return True
        
        cache_time = os.path.getmtime(self.cache_file)
        
        # Vérifier si des images ont été modifiées après la création du cache
        for person_name in os.listdir(self.images_folder):
            person_dir = os.path.join(self.images_folder, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            if os.path.getmtime(person_dir) > cache_time:
                return True
            
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                if os.path.getmtime(image_path) > cache_time:
                    return True
        
        return False
    
    def load_known_faces(self):
        """Charge les visages connus depuis le cache ou les images"""
        if not self.need_to_reload():
            print("Chargement des visages connus depuis le cache...")
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"Cache chargé: {len(self.known_face_encodings)} visages de {len(set(self.known_face_names))} personnes")
                return
            except Exception as e:
                print(f"Erreur lors du chargement du cache: {e}")
        
        print("Encodage des visages connus (première fois ou mise à jour)...")
        self.encode_known_faces_from_folders()
        self.save_cache()
    
    def encode_known_faces_from_folders(self):
        """Encode les visages depuis les dossiers d'images"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        for person_name in os.listdir(self.images_folder):
            print(f"Traitement: {person_name}")
            person_dir = os.path.join(self.images_folder, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    encodings = detect_and_encode(image_rgb)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(person_name)
    
    def save_cache(self):
        """Sauvegarde le cache des visages connus"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Cache sauvegardé: {len(self.known_face_encodings)} encodages")

class IntruderCollector:
    def __init__(self, base_folder="Intrus_temp"):
        self.base_folder = base_folder
        self.intruder_images = []  # Liste des images d'intrus capturées
        self.intruder_encodings = []  # Encodages correspondants
        self.capture_counter = 0
        
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
    
    def add_intruder(self, frame, encoding):
        """Ajoute un intrus à la collection temporaire"""
        self.capture_counter += 1
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"temp_intruder_{self.capture_counter:04d}_{timestamp}.jpg"
        filepath = os.path.join(self.base_folder, filename)
        
        # Sauvegarder l'image temporairement
        cv2.imwrite(filepath, frame)
        
        # Stocker les données
        self.intruder_images.append(filepath)
        self.intruder_encodings.append(encoding)
        
        # Sauvegarder dans MongoDB avec un ID temporaire
        _, buffer = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        
        temp_data = {
            "temp_id": f"temp_{self.capture_counter:04d}",
            "timestamp": timestamp,
            "image_base64": img_base64,
            "classified": False
        }
        collection.insert_one(temp_data)
        
        print(f"Intrus capturé: {filename}")
        return filepath

class IntruderClassifier:
    def __init__(self, temp_folder="Intrus_temp", final_folder="Intrus"):
        self.temp_folder = temp_folder
        self.final_folder = final_folder
        self.classified_folder = os.path.join(final_folder, "Classes")
        
        if not os.path.exists(self.final_folder):
            os.makedirs(self.final_folder)
        if not os.path.exists(self.classified_folder):
            os.makedirs(self.classified_folder)
    
    def _cleanup_temp_folder(self):
        """Nettoie le dossier temporaire après classification"""
        try:
            for file in os.listdir(self.temp_folder):
                file_path = os.path.join(self.temp_folder, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Erreur lors de la suppression de {file_path}: {e}")
            print("Nettoyage du dossier temporaire terminé")
        except Exception as e:
            print(f"Erreur lors du nettoyage du dossier temporaire: {e}")
    
    def _get_next_intruder_number(self):
        """Détermine le prochain numéro d'intrus disponible"""
        if not os.path.exists(self.classified_folder):
            return 1
            
        existing_folders = []
        for folder in os.listdir(self.classified_folder):
            if folder.startswith("Intrus_"):
                try:
                    num = int(folder.split("_")[1])
                    existing_folders.append(num)
                except (ValueError, IndexError):
                    continue
                    
        return max(existing_folders, default=0) + 1
    
    def _load_existing_intruders(self):
        """Charge les encodages des intrus déjà classifiés"""
        existing_encodings = []
        existing_folders = []
        
        if not os.path.exists(self.classified_folder):
            return [], []
            
        for folder in sorted(os.listdir(self.classified_folder)):
            if folder.startswith("Intrus_"):
                folder_path = os.path.join(self.classified_folder, folder)
                if os.path.isdir(folder_path):
                    existing_folders.append(folder)
                    # Prendre la première image du dossier comme référence
                    for image_name in os.listdir(folder_path):
                        image_path = os.path.join(folder_path, image_name)
                        image = cv2.imread(image_path)
                        if image is not None:
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            encodings = detect_and_encode(image_rgb)
                            if encodings:
                                existing_encodings.append(encodings[0])
                                break  # On ne prend que la première image
        
        return existing_encodings, existing_folders
    
    def classify_intruders(self, encodings_list, image_paths):
        """Classifie tous les intrus capturés à la fin"""
        if not encodings_list:
            print("Aucun intrus à classifier")
            return
        
        print(f"\n=== CLASSIFICATION DE {len(encodings_list)} INTRUS ===")
        
        # Charger les intrus existants
        existing_encodings, existing_folders = self._load_existing_intruders()
        if existing_encodings:
            print(f"Chargement de {len(existing_encodings)} intrus existants")
        
        # Comparer d'abord avec les intrus existants
        new_encodings = []
        new_image_paths = []
        assigned_to_existing = {}  # Pour suivre les attributions aux intrus existants
        
        for idx, encoding in enumerate(encodings_list):
            if existing_encodings:
                # Calculer les distances avec les intrus existants
                distances = np.linalg.norm(np.array(existing_encodings) - encoding, axis=1)
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                
                if min_distance < 0.6:  # Même seuil que pour la reconnaissance
                    existing_folder = existing_folders[min_distance_idx]
                    if existing_folder not in assigned_to_existing:
                        assigned_to_existing[existing_folder] = []
                    assigned_to_existing[existing_folder].append(idx)
                    continue
            
            new_encodings.append(encoding)
            new_image_paths.append(image_paths[idx])
        
        # Traiter d'abord les images assignées aux intrus existants
        for folder, indices in assigned_to_existing.items():
            print(f"Ajout de {len(indices)} images à {folder}")
            self._move_intruder_images(indices, folder, image_paths)
        
        # Traiter les nouveaux intrus
        if new_encodings:
            print(f"\nClassification de {len(new_encodings)} nouveaux intrus")
            
            # Utiliser DBSCAN pour le clustering des nouveaux intrus
            clustering = DBSCAN(eps=0.6, min_samples=1, metric='euclidean')
            cluster_labels = clustering.fit_predict(np.array(new_encodings))
            
            # Organiser par clusters
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)
            
            print(f"Trouvé {len(clusters)} nouveaux groupes d'intrus")
            
            # Obtenir le prochain numéro d'intrus disponible
            intruder_counter = self._get_next_intruder_number()
            
            # Créer les nouveaux dossiers d'intrus
            for cluster_id, indices in clusters.items():
                if cluster_id == -1:  # Outliers/bruit
                    for idx in indices:
                        intruder_name = f"Intrus_{intruder_counter:03d}"
                        self._move_intruder_images([idx], intruder_name, new_image_paths)
                        intruder_counter += 1
                else:  # Cluster normal
                    intruder_name = f"Intrus_{intruder_counter:03d}"
                    self._move_intruder_images(indices, intruder_name, new_image_paths)
                    intruder_counter += 1
        
        # Nettoyer le dossier temporaire
        self._cleanup_temp_folder()
        
        print(f"=== CLASSIFICATION TERMINÉE ===")
    
    def _move_intruder_images(self, indices, intruder_name, image_paths):
        """Déplace les images d'un intrus vers son dossier final"""
        intruder_folder = os.path.join(self.classified_folder, intruder_name)
        os.makedirs(intruder_folder, exist_ok=True)
        
        # Compter les images existantes dans le dossier
        existing_images = len([f for f in os.listdir(intruder_folder) if f.endswith('.jpg')])
        
        print(f"{'Ajout dans' if existing_images > 0 else 'Création du dossier'}: {intruder_name} ({len(indices)} images)")
        
        for i, idx in enumerate(indices):
            old_path = image_paths[idx]
            if os.path.exists(old_path):
                # Nouveau nom avec numérotation
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                new_filename = f"{intruder_name}_{existing_images + i + 1:03d}_{timestamp}.jpg"
                new_path = os.path.join(intruder_folder, new_filename)
                
                # Déplacer le fichier
                os.rename(old_path, new_path)
                
                # Lire l'image et l'encoder en base64 pour MongoDB
                image = cv2.imread(new_path)
                _, buffer = cv2.imencode(".jpg", image)
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                
                # Enregistrer dans MongoDB
                intruder_data = {
                    "intruder_id": intruder_name,
                    "image_number": existing_images + i + 1,
                    "timestamp": timestamp,
                    "image_path": new_path,
                    "image_base64": img_base64,
                    "classified": True
                }
                collection.insert_one(intruder_data)

# Function to detect and encode faces
def detect_and_encode(image):
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0)
                encoding = resnet(face_tensor).detach().numpy().flatten()
                faces.append(encoding)
            return faces
    return []

def recognize_faces(known_encodings, known_names, test_encodings, frame, intruder_collector, threshold=0.6):
    """Reconnaissance avec collection temporaire des intrus"""
    recognized_names = []
    
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            # Intrus détecté - ajouter à la collection temporaire si disponible
            if intruder_collector is not None:
                filepath = intruder_collector.add_intruder(frame, test_encoding)
                recognized_names.append('INTRUS CAPTURÉ')
            else:
                recognized_names.append('INTRUS DÉTECTÉ')
    
    return recognized_names

# === PROGRAMME PRINCIPAL ===
def main():
    print("=== SYSTÈME DE RECONNAISSANCE FACIALE AVEC CLASSIFICATION INTELLIGENTE ===")
    
    # 1. Chargement des visages connus (avec cache)
    print("\n1. Chargement des visages connus...")
    known_faces_manager = KnownFacesManager()
    known_faces_manager.load_known_faces()
    
    known_encodings = np.array(known_faces_manager.known_face_encodings)
    known_names = known_faces_manager.known_face_names
    
    # 2. Initialisation du collecteur d'intrus
    print("\n2. Initialisation du système de capture...")
    intruder_collector = IntruderCollector()
    
    # 3. Capture vidéo en temps réel
    print("\n3. Démarrage de la capture vidéo...")
    print("Appuyez sur 'q' pour arrêter et lancer la classification")
    
    cap = cv2.VideoCapture(0)
    threshold = 0.6
    last_capture_time = 0
    capture_interval = 2  # Capturer un intrus toutes les 2 secondes maximum
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        test_face_encodings = detect_and_encode(frame_rgb)
        
        current_time = time.time()
        
        if test_face_encodings and len(known_encodings) > 0:
            # Déterminer si on doit capturer (respecter l'intervalle)
            should_capture = current_time - last_capture_time > capture_interval
            
            names = recognize_faces(
                known_encodings, 
                known_names, 
                test_face_encodings, 
                frame, 
                intruder_collector if should_capture else None,
                threshold
            )
            
            # Mettre à jour le timer si un intrus a été capturé
            if should_capture and ('INTRUS CAPTURÉ' in names):
                last_capture_time = current_time
            
            # Affichage des résultats
            boxes, _ = mtcnn.detect(frame_rgb)
            if boxes is not None:
                for name, box in zip(names, boxes):
                    if box is not None:
                        (x1, y1, x2, y2) = map(int, box)
                        
                        color = (0, 0, 255) if 'INTRUS' in name else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, name, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        # Affichage des statistiques
        cv2.putText(frame, f"Intrus captures: {intruder_collector.capture_counter}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Personnes connues: {len(set(known_names))}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Capture des Intrus - Appuyez sur Q pour classifier', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 4. Classification finale des intrus
    print("\n4. Classification des intrus capturés...")
    classifier = IntruderClassifier()
    classifier.classify_intruders(intruder_collector.intruder_encodings, intruder_collector.intruder_images)
    
    print("\n=== PROGRAMME TERMINÉ ===")

if __name__ == "__main__":
    main()