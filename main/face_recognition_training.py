import os
import numpy as np
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from django.conf import settings
from .models import Employee, UploadedPhoto

class FaceRecognitionTraining:
    def __init__(self):
        self.target_size = (160, 160)
        self.detector = MTCNN()
        self.embedder = FaceNet()
        self.norm_encoder = Normalizer(norm='l2')

    def preprocess_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Unable to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(img)
        if not detections:
            raise Exception("No faces detected in the image.")

        x, y, w, h = detections[0]['box']
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        return face_img

    def get_embedding(self, face_img):
        face_img = np.asarray(face_img, dtype=np.float32)
        face_img = np.expand_dims(face_img, axis=0)
        return self.embedder.embeddings(face_img)[0]

    def load_data_from_django(self):
        X = []
        Y = []
        for employee in Employee.objects.all():
            photos = UploadedPhoto.objects.filter(employee=employee)
            for photo in photos:
                try:
                    img_path = os.path.join(settings.MEDIA_ROOT, str(photo.photo))
                    face_img = self.preprocess_image(img_path)
                    embedding = self.get_embedding(face_img)
                    X.append(embedding)
                    Y.append(f"{employee.name} {employee.role}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        return np.array(X), np.array(Y)

    def train_model(self):
        X, Y = self.load_data_from_django()

        # Save labels to file
        unique_labels = sorted(set(Y))
        labels_file = os.path.join(settings.MEDIA_ROOT, 'face_labels.txt')
        with open(labels_file, 'w') as file:
            file.write("\n".join(unique_labels))

        # Save embeddings and labels
        output_file = os.path.join(settings.MEDIA_ROOT, 'faces_embeddings_done.npz')
        np.savez_compressed(output_file, embeddings=X, labels=Y)

        # Prepare data for model training
        encoder = LabelEncoder()
        Y_encoded = encoder.fit_transform(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, shuffle=True, random_state=17)

        X_train_norm = self.norm_encoder.fit_transform(X_train)
        X_test_norm = self.norm_encoder.transform(X_test)

        num_classes = len(unique_labels)
        Y_train_categorical = to_categorical(Y_train, num_classes)
        Y_test_categorical = to_categorical(Y_test, num_classes)

        # Define and train the model
        model = Sequential([
            Dense(512, input_shape=(X_train_norm.shape[1],), activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(
            X_train_norm, Y_train_categorical,
            epochs=100, batch_size=32,
            validation_data=(X_test_norm, Y_test_categorical)
        )

        # Save the model
        model_path = os.path.join(settings.MEDIA_ROOT, 'mtcnn_facenet_ann_model.h5')
        model.save(model_path)

        return model_path, labels_file, output_file

def train_face_recognition_model():
    trainer = FaceRecognitionTraining()
    model_path, labels_file, embeddings_file = trainer.train_model()
    return model_path, labels_file, embeddings_file
