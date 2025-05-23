import os
import cv2
import numpy as np
import torch
import pymongo
import time
import base64
from facenet_pytorch import InceptionResnetV1, MTCNN

# Connexion à MongoDB (change l'URI si tu utilises un serveur distant)
client = pymongo.MongoClient("mongodb+srv://maherbouchrara:maherbouchrara@iotproject.6zy5h.mongodb.net/?retryWrites=true&w=majority&appName=IotProject")
db = client["IotProject"]  # Nom de la base de données
collection = db["Intruders"]  # Collection pour stocker les intrus

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

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

# Function to encode specific images with predefined names
# def encode_known_faces(known_faces):
#     known_face_encodings = []
#     known_face_names = []

#     for name, image_path in known_faces.items():
#         known_image = cv2.imread(image_path)
#         if known_image is not None:
#             known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
#             encodings = detect_and_encode(known_image_rgb)
#             if encodings:
#                 known_face_encodings.append(encodings[0])  # Assuming one face per image
#                 known_face_names.append(name)

#     return known_face_encodings, known_face_names

def encode_known_faces_from_folders(base_dir):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(base_dir):
        print(person_name)
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

# Define known faces with explicit names
known_faces = {
    "louay njeh": "images/Asif.jpg",
    "Maram": "images/maram.jpg",
    "Nawres": "images/nawres.jpg",
    "Said": "images/said.jpg"
}

# Encode known faces
# known_face_encodings, known_face_names = encode_known_faces(known_faces)
known_face_encodings, known_face_names = encode_known_faces_from_folders("Images1")


# Creation of the intruder folder
intruder_folder = "Intrus"
if not os.path.exists(intruder_folder):
    os.makedirs(intruder_folder)

# Timer de capture
last_capture_time = 0  # Initialiser le timer
last_capture_time = time.time()

# Function to recognize faces
def recognize_faces(known_encodings, known_names, test_encodings,frame , intruder_folder, threshold=0.6):
    global last_capture_time
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            recognized_names.append('Not Recognized')

            # Capture d'écran toutes les 5 secondes pour un intrus
            if time.time() - last_capture_time >= 5:
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                filepath = os.path.join(intruder_folder, f"Intrus_{timestamp}.jpg")
                cv2.imwrite(filepath, frame)
                print(f" Intrus détecté ! Capture enregistrée : {filepath}")
                last_capture_time = time.time()  # Mise à jour du timer

                 # Convertir l’image en base64 pour MongoDB
                _, buffer = cv2.imencode(".jpg", frame)
                img_base64 = base64.b64encode(buffer).decode("utf-8")

                                # Insérer dans MongoDB
                intruder_data = {
                    "timestamp": timestamp,
                    "image_base64": img_base64
                }
                collection.insert_one(intruder_data)

            # print('hey')
    return recognized_names

# Start video capture
cap = cv2.VideoCapture(0)
threshold = 0.6

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    test_face_encodings = detect_and_encode(frame_rgb)

    if test_face_encodings and known_face_encodings:
        names = recognize_faces(np.array(known_face_encodings), known_face_names, test_face_encodings, frame, intruder_folder,threshold)
        for name, box in zip(names, mtcnn.detect(frame_rgb)[0]):
            if box is not None:
                (x1, y1, x2, y2) = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
