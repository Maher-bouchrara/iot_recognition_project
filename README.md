# Face Recognition & Intruder Classification System

This project is a real-time face recognition and intruder classification system using deep learning and clustering. It detects faces from a webcam, recognizes known individuals, and captures/classifies unknown (intruder) faces. All data is stored in MongoDB, and the system uses a cache for known faces to speed up recognition.

## Features
- Real-time face detection and recognition using webcam
- Automatic caching of known faces for fast startup
- Intruder (unknown face) capture and temporary storage
- Final clustering and classification of intruders using DBSCAN
- All images and metadata stored in MongoDB

## Requirements
- **Python 3.12 or lower is required** (Python 3.13 is not supported by Pillow/facenet-pytorch)
- Windows OS (tested)
- Webcam

## Installation

1. **Clone the repository or copy the project files**

2. **Install dependencies**

Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

If you use a GPU, you may want to install a CUDA-enabled version of `torch` (see https://pytorch.org/get-started/locally/).

3. **MongoDB Setup**

- The script uses a MongoDB Atlas cloud database. Update the connection string in `face_rec_f.py` if you want to use your own database.
- No local MongoDB installation is required if you use the provided connection string.

4. **Prepare Known Faces**

- Place images of known people in the `Images1/` directory.
- Each person should have their own subfolder (e.g., `Images1/JohnDoe/` with images inside).

5. **Run the Program**

In the project directory, run:

```bash
python face_rec_f.py
```

- The webcam will open and start recognizing faces.
- Press `q` to stop the video and trigger intruder classification.

## Project Structure
- `face_rec_f.py` : Main script for recognition and classification
- `Images1/` : Folder containing subfolders of known people and their images
- `Intrus_temp/` : Temporary storage for captured intruder images (auto-created)
- `Intrus/Classes/` : Final classified intruder images (auto-created)
- `requirements.txt` : Python dependencies
- `known_faces_cache.pkl` : Cache file for known faces (auto-generated)

## Notes
- The system uses `facenet-pytorch` for face detection and embedding.
- Intruder images and metadata are stored in MongoDB for later review.
- You can adjust thresholds and clustering parameters in the script for your needs.

## Troubleshooting
- If you get errors about missing DLLs or `cv2`, ensure you have the correct version of OpenCV and Python.
- For GPU acceleration, ensure you have the correct CUDA drivers and PyTorch version.
- If MongoDB connection fails, check your internet connection and credentials.
- If you get errors installing Pillow or facenet-pytorch, make sure you are using Python 3.12 or lower.

## License
This project is for educational and research purposes.
