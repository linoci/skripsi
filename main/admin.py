from django.contrib import admin
from .models import Employee, UploadedPhoto, FaceLoginLog
# Register your models here.'

# Kelas Admin untuk model Employee
@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    list_display = ('name', 'role', 'email', 'address', 'username', 'satker', 'password', 'jabatan', 'birthday', 'nip','phone')  # Kolom yang ditampilkan di admin
    search_fields = ('name', 'email', 'role')  # Kolom yang dapat dicari
    list_filter = ('role',)  # Filter berdasarkan role

# Kelas Admin untuk model UploadedPhoto
@admin.register(UploadedPhoto)
class UploadedPhotoAdmin(admin.ModelAdmin):
    list_display = ('employee', 'photo', 'upload_time')  # Kolom yang ditampilkan di admin
    search_fields = ('employee__name',)  # Pencarian berdasarkan nama employee

@admin.register(FaceLoginLog)
class FaceLoginLogAdmin(admin.ModelAdmin):
    list_display = ('name', 'role', 'timestamp', 'ip_address')
    ordering = ('-timestamp',)
