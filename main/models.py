from django.db import models
from django.db.models.signals import post_delete, pre_save
from django.dispatch import receiver
import os
import shutil
from django.contrib.auth.hashers import make_password
from django.utils import timezone

def employee_photo_path(instance, filename):
    """
    Fungsi untuk membuat jalur penyimpanan file gambar secara dinamis.
    Jalur disesuaikan berdasarkan role dan nama pegawai.
    """
    return os.path.join(instance.employee.role, instance.employee.name, filename)

class Employee(models.Model):
    ROLE_CHOICES = [
        ('supervisor', 'Supervisor'),
        ('employee', 'Employee'),
    ]
    
    name = models.CharField(max_length=100)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    email = models.EmailField()
    address = models.CharField(max_length=255)
    username = models.CharField(max_length=50, unique=True)
    password = models.CharField(max_length=128)
    satker = models.CharField(max_length=50)  # Perbaiki di sini
    jabatan = models.CharField(max_length=50)
    birthday = models.DateField()
    nip = models.CharField(max_length=18)
    phone = models.CharField(max_length=15)
    is_trained = models.BooleanField(default=False)
    last_login = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['username']),
            models.Index(fields=['role']),
        ]

# Signal to hash password before saving
@receiver(pre_save, sender=Employee)
def hash_password(sender, instance, **kwargs):
    # Only hash if password changed (not already hashed)
    if instance.password and not instance.password.startswith('pbkdf2_sha256$'):
        instance.password = make_password(instance.password)

class UploadedPhoto(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    photo = models.ImageField(upload_to=employee_photo_path)
    upload_time = models.DateTimeField(auto_now_add=True)

# Signal untuk menghapus file dan folder saat objek dihapus
@receiver(post_delete, sender=UploadedPhoto)
def delete_photo_file(sender, instance, **kwargs):
    """
    Signal yang akan memanggil saat objek UploadedPhoto dihapus.
    Ini akan menghapus file gambar dari sistem file dan jika folder kosong, akan menghapus folder juga.
    """
    # Hapus file gambar
    if instance.photo:
        if os.path.isfile(instance.photo.path):
            try:
                os.remove(instance.photo.path)  # Hapus file gambar
            except Exception as e:
                print(f"Error saat menghapus file: {e}")
        
        # Cek apakah folder kosong, jika iya, hapus foldernya
        folder = os.path.dirname(instance.photo.path)
        try:
            # Hanya hapus folder jika kosong
            if os.path.isdir(folder) and not os.listdir(folder):
                shutil.rmtree(folder)  # Hapus folder jika kosong
        except Exception as e:
            print(f"Error saat menghapus folder: {e}")

class FaceLoginLog(models.Model):
    session_id = models.CharField(max_length=100, null=True, blank=True)
    name = models.CharField(max_length=255)
    role = models.CharField(max_length=100)
    timestamp = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    photo = models.ImageField(upload_to='login_photos/', null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.role}) at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
