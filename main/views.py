import subprocess
import cv2
import numpy as np
import time
import os
import webbrowser
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.views.decorators import gzip
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from django.conf import settings
from keras_facenet import FaceNet
from flask import Flask, render_template, Response
from .forms import EmployeeForm, PhotoUploadForm, LoginForm
from .models import Employee, UploadedPhoto
from django.contrib import messages
from django.contrib.auth.hashers import make_password, check_password
from django.db import transaction
from django.db.models import Exists, OuterRef
from django.views.decorators.http import require_POST
from .face_recognition_training import train_face_recognition_model
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from .models import FaceLoginLog
import json
from datetime import datetime, timedelta
from django.utils import timezone
from django.db.models.functions import ExtractMonth, ExtractYear

# Load model dan label
model = load_model(os.path.join(settings.BASE_DIR, '../utama', 'mtcnn_facenet_ann_model.h5'))
with open(os.path.join(settings.BASE_DIR, '../utama', 'face_labels.txt'), 'r') as f:
    labels = [line.strip() for line in f.readlines()]

encoder = LabelEncoder()
encoder.fit(labels)

# Load detektor
embedder = FaceNet()
img_detector = MTCNN()
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Variabel global
prev_bbox = None
last_label = None
label_start_time = None
required_label_duration = 3

def home(request):
    return render(request, 'home.html')

def login_view(request):
    return render(request, 'login.html')

def super_view(request):
    return render(request, 'supervisor.html')

def pegawai_view(request):
    logs = FaceLoginLog.objects.filter(role='employee').order_by('-timestamp')
    return render(request, 'pegawai.html', {'logs': logs})

def get_embedding(face_img):
    face_img = np.asarray(face_img, dtype=np.float32)
    face_img = np.expand_dims(face_img, axis=0)
    return embedder.embeddings(face_img)[0]

def predict_class(embedding, model, encoder, threshold=0.9):
    test_embedding = np.expand_dims(embedding, axis=0)
    predict_proba = model.predict(test_embedding)[0]
    predicted_class = np.argmax(predict_proba)
    confidence_score = predict_proba[predicted_class]
    if confidence_score < threshold:
        return "unknown", 0.0
    return encoder.inverse_transform([predicted_class])[0], confidence_score * 100

def load_and_preprocess_image_from_frame(frame):
    t_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = t_img.shape
    scale = min(480 / width, 480 / height)
    resized_img = cv2.resize(t_img, (int(width * scale), int(height * scale)))
    detections = img_detector.detect_faces(resized_img)
    if detections:
        x, y, w, h = detections[0]['box']
        return resized_img, (x, y, w, h)
    return resized_img, None

def crop_and_get_embedding(image, bbox, target_size=(160, 160)):
    x, y, w, h = bbox
    face_img = image[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, target_size)
    return get_embedding(face_img)

def detect_smile_and_eyes(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
    return len(smiles) > 0, len(eyes) >= 2

def is_head_moving(current_bbox, prev_bbox, threshold=10):
    if prev_bbox is None:
        return False
    x1, y1, _, _ = current_bbox
    x2, y2, _, _ = prev_bbox
    movement = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return movement > threshold

def save_login_photo(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.jpg"
    filepath = os.path.join(settings.MEDIA_ROOT, 'login_photos', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, frame)
    return f"login_photos/{filename}"

@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(gen_frames(request), content_type='multipart/x-mixed-replace; boundary=frame')

def gen_frames(request):
    global prev_bbox, last_label, label_start_time
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_img, bbox = load_and_preprocess_image_from_frame(frame)
        if bbox:
            x, y, w, h = bbox
            face_img = resized_img[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            head_moved = is_head_moving(bbox, prev_bbox, 10)
            prev_bbox = bbox

            smile_detected, eyes_detected = detect_smile_and_eyes(cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

            if smile_detected and eyes_detected and head_moved:
                embedding = crop_and_get_embedding(resized_img, bbox)
                predicted_label, confidence = predict_class(embedding, model, encoder)

                current_time = time.time()
                if predicted_label == last_label:
                    if current_time - label_start_time >= required_label_duration:
                        print(f"Label terverifikasi: {predicted_label}")
                        ip_address = request.META.get('REMOTE_ADDR')
                        latitude = request.session.get('latitude')
                        longitude = request.session.get('longitude')
                        photo_path = save_login_photo(frame, predicted_label.replace(" ", "_"))

                        session_id = request.session.session_key
                        if not session_id:
                            request.session.create()
                            session_id = request.session.session_key

                        if "supervisor" in predicted_label.lower():
                            role = 'supervisor'
                        else:
                            role = 'employee'

                        FaceLoginLog.objects.create(
                            name=predicted_label,
                            role=role,
                            ip_address=ip_address,
                            latitude=latitude,
                            longitude=longitude,
                            photo=photo_path,
                            session_id=session_id
                        )
                else:
                    last_label = predicted_label
                    label_start_time = current_time

                cv2.putText(frame, f"{predicted_label} ({confidence:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                cv2.putText(frame, "Liveness detection failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def check_auth(request):
    session_id = request.session.session_key
    if not session_id:
        return JsonResponse({'redirect': None})

    recent_log = FaceLoginLog.objects.filter(
        session_id=session_id,
        timestamp__gte=timezone.now() - timedelta(minutes=2)
    ).order_by('-timestamp').first()

    if recent_log:
        if recent_log.role == 'supervisor':
            return JsonResponse({'redirect': '/supervisor/'})
        elif recent_log.role == 'employee':
            return JsonResponse({'redirect': '/pegawai/'})

    return JsonResponse({'redirect': None})

@csrf_exempt
def save_location(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        request.session['latitude'] = data.get('latitude')
        request.session['longitude'] = data.get('longitude')
        return JsonResponse({'status': 'ok'})
    return JsonResponse({'status': 'invalid'}, status=400)

# View tambah pegawai
def add_employee(request):
    if request.method == 'POST':
        employee_form = EmployeeForm(request.POST)
        photo_form = PhotoUploadForm(request.POST, request.FILES)

        if employee_form.is_valid() and photo_form.is_valid():
            try:
                with transaction.atomic():
                    employee = employee_form.save(commit=False)
                    employee.password = make_password(employee.password)
                    employee.save()

                    photos = request.FILES.getlist('photos')
                    for photo in photos:
                        role_folder = os.path.join(settings.MEDIA_ROOT, employee.role)
                        user_folder = os.path.join(role_folder, employee.name)

                        if not os.path.exists(user_folder):
                            os.makedirs(user_folder)

                        file_path = os.path.join(user_folder, photo.name)
                        with open(file_path, 'wb+') as destination:
                            for chunk in photo.chunks():
                                destination.write(chunk)

                        UploadedPhoto.objects.create(
                            employee=employee,
                            photo=f'{employee.role}/{employee.name}/{photo.name}'
                        )

                    messages.success(request, "Data pegawai berhasil disimpan.")
                    return redirect('index')

            except Exception as e:
                messages.error(request, f"Terjadi kesalahan saat menyimpan data: {str(e)}")
        else:
            print(f"Employee Form Errors: {employee_form.errors}")
            print(f"Photo Form Errors: {photo_form.errors}")
            messages.error(request, "Data tidak valid. Silakan periksa kembali form.")

    else:
        employee_form = EmployeeForm()
        photo_form = PhotoUploadForm()

    return render(request, 'add_user.html', {
        'employee_form': employee_form,
        'photo_form': photo_form
    })

# View daftar data pegawai untuk pelatihan
def train_data(request):
    employees = Employee.objects.all().annotate(
        has_photos=Exists(UploadedPhoto.objects.filter(employee=OuterRef('pk')))
    )
    return render(request, "train_data.html", {'employees': employees})

# View untuk memproses pelatihan model
@require_POST
def train_model_view(request):
    try:
        model_path, labels_file, embeddings_file = train_face_recognition_model()
        Employee.objects.update(is_trained=True)
        return JsonResponse({
            'status': 'success',
            'message': 'Model training completed successfully.',
            'model_path': model_path,
            'labels_file': labels_file,
            'embeddings_file': embeddings_file
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

# View untuk memuat model yang sudah dilatih
def load_model_view(request):
    try:
        model_path = os.path.join(settings.MEDIA_ROOT, 'mtcnn_facenet_ann_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

        model = load_model(model_path)
        labels_path = os.path.join(settings.MEDIA_ROOT, 'face_labels.txt')
        with open(labels_path, 'r') as file:
            labels = [line.strip() for line in file]

        return JsonResponse({'status': 'success', 'message': 'Model and labels loaded successfully.'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

#def log_view(request):
    #logs = FaceLoginLog.objects.all().order_by('-timestamp')
    #return render(request, 'log_login.html', {'logs': logs})
def log_view(request):
    name = request.GET.get('name')
    is_print = request.GET.get('print')

    logs = FaceLoginLog.objects.all().order_by('-timestamp')
    if name:
        logs = logs.filter(name__icontains=name)

    if is_print:
        return render(request, 'log_login_print.html', {
            'logs': logs,
            'name': name
        })

    return render(request, 'log_login.html', {
        'logs': logs,
        'name': name
    })
def divisi(request):
    return render(request, 'divisi.html')
