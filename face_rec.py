import os
import cv2
import numpy as np
import torch
import pymongo
import time
import base64
import json
from facenet_pytorch import InceptionResnetV1, MTCNN

# Connexion à MongoDB
client = pymongo.MongoClient("mongodb+srv://maherbouchrara:maherbouchrara@iotproject.6zy5h.mongodb.net/?retryWrites=true&w=majority&appName=IotProject")
db = client["IotProject"]
collection = db["Intruders"]

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Classe pour gérer les intrus
class IntruderManager:
    def __init__(self, base_intruder_folder="Intrus"):
        self.base_intruder_folder = base_intruder_folder
        self.intruder_encodings = {}  # {intruder_id: [list_of_encodings]}
        self.intruder_counter = 0
        self.encodings_file = os.path.join(base_intruder_folder, "intruder_encodings.json")
        
        # Créer le dossier principal des intrus
        if not os.path.exists(base_intruder_folder):
            os.makedirs(base_intruder_folder)
        
        # Charger les encodages existants
        self.load_intruder_encodings()
    
    def save_intruder_encodings(self):
        """Sauvegarder les encodages des intrus dans un fichier JSON"""
        data = {
            "intruder_counter": self.intruder_counter,
            "intruder_encodings": {k: [enc.tolist() for enc in v] for k, v in self.intruder_encodings.items()}
        }
        with open(self.encodings_file, 'w') as f:
            json.dump(data, f)
    
    def load_intruder_encodings(self):
        """Charger les encodages des intrus depuis le fichier JSON"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'r') as f:
                    data = json.load(f)
                self.intruder_counter = data.get("intruder_counter", 0)
                self.intruder_encodings = {
                    k: [np.array(enc) for enc in v] 
                    for k, v in data.get("intruder_encodings", {}).items()
                }
                print(f"Chargé {len(self.intruder_encodings)} intrus existants")
            except Exception as e:
                print(f"Erreur lors du chargement des encodages: {e}")
    
    def find_matching_intruder(self, new_encoding, threshold=0.6):
        """Trouve si l'encodage correspond à un intrus existant"""
        for intruder_id, encodings_list in self.intruder_encodings.items():
            for existing_encoding in encodings_list:
                distance = np.linalg.norm(existing_encoding - new_encoding)
                if distance < threshold:
                    return intruder_id, float(distance)  # Conversion explicite en float
        return None, float('inf')
    
    def add_intruder_image(self, frame, encoding):
        """Ajoute une image d'intrus et gère la classification"""
        # Chercher si cet intrus existe déjà
        matching_intruder, distance = self.find_matching_intruder(encoding)
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        if matching_intruder:
            # Intrus existant - ajouter à son dossier
            intruder_folder = os.path.join(self.base_intruder_folder, matching_intruder)
            print(f"Intrus reconnu: {matching_intruder} (distance: {distance:.3f})")
        else:
            # Nouvel intrus - créer un nouveau dossier
            self.intruder_counter += 1
            intruder_id = f"Intrus_{self.intruder_counter:03d}"
            intruder_folder = os.path.join(self.base_intruder_folder, intruder_id)
            
            if not os.path.exists(intruder_folder):
                os.makedirs(intruder_folder)
            
            # Initialiser la liste d'encodages pour ce nouvel intrus
            self.intruder_encodings[intruder_id] = []
            matching_intruder = intruder_id
            distance = 0.0  # Distance 0 pour un nouvel intrus
            print(f"Nouvel intrus détecté: {intruder_id}")
        
        # Ajouter l'encodage à la liste de cet intrus (max 5 encodages par intrus)
        if len(self.intruder_encodings[matching_intruder]) < 5:
            self.intruder_encodings[matching_intruder].append(encoding)
            self.save_intruder_encodings()
        
        # Sauvegarder l'image
        filepath = os.path.join(intruder_folder, f"{matching_intruder}_{timestamp}.jpg")
        cv2.imwrite(filepath, frame)
        
        # Convertir l'image en base64 pour MongoDB
        _, buffer = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        
        # Insérer dans MongoDB
        intruder_data = {
            "intruder_id": matching_intruder,
            "timestamp": timestamp,
            "image_base64": img_base64,
            "distance": float(distance) if matching_intruder else 0.0
        }
        collection.insert_one(intruder_data)
        
        return matching_intruder, filepath

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

def encode_known_faces_from_folders(base_dir):
    known_face_encodings = []
    known_face_names = []
    
    for person_name in os.listdir(base_dir):
        print(f"Chargement des images de: {person_name}")
        person_dir = os.path.join(base_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                encodings = detect_and_encode(image_rgb)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name)
    
    return known_face_encodings, known_face_names

# Function to recognize faces with improved intruder management
def recognize_faces(known_encodings, known_names, test_encodings, frame, intruder_manager, threshold=0.6):
    recognized_names = []
    
    for i, test_encoding in enumerate(test_encodings):
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            # Intrus détecté - utiliser le gestionnaire d'intrus
            intruder_id, filepath = intruder_manager.add_intruder_image(frame, test_encoding)
            recognized_names.append(f'INTRUS: {intruder_id}')
            print(f"Image d'intrus sauvegardée: {filepath}")
    
    return recognized_names

# Initialisation
print("Chargement des visages connus...")
known_face_encodings, known_face_names = encode_known_faces_from_folders("Images1")
print(f"Chargé {len(known_face_encodings)} visages connus de {len(set(known_face_names))} personnes")

# Initialiser le gestionnaire d'intrus
intruder_manager = IntruderManager()

# Timer pour éviter la capture trop fréquente
last_capture_time = {}  # {intruder_id: last_time}

# Start video capture
cap = cv2.VideoCapture(0)
threshold = 0.6

print("Démarrage de la reconnaissance faciale...")
print("Appuyez sur 'q' pour quitter")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    test_face_encodings = detect_and_encode(frame_rgb)
    
    if test_face_encodings and known_face_encodings:
        names = recognize_faces(
            np.array(known_face_encodings), 
            known_face_names, 
            test_face_encodings, 
            frame, 
            intruder_manager, 
            threshold
        )
        
        # Afficher les résultats sur l'image
        boxes, _ = mtcnn.detect(frame_rgb)
        if boxes is not None:
            for name, box in zip(names, boxes):
                if box is not None:
                    (x1, y1, x2, y2) = map(int, box)
                    
                    # Couleur différente pour les intrus
                    if 'INTRUS' in name:
                        color = (0, 0, 255)  # Rouge pour les intrus
                    else:
                        color = (0, 255, 0)  # Vert pour les personnes connues
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, name, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    # Afficher des statistiques
    cv2.putText(frame, f"Intrus classes: {intruder_manager.intruder_counter}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Personnes connues: {len(set(known_face_names))}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Face Recognition - Classification des Intrus', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Système arrêté.")
print(f"Total d'intrus classés: {intruder_manager.intruder_counter}")