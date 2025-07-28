from django.urls import path
from . import views
from .views import add_employee, train_model_view
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.home, name='home'),
    path('', views.home, name='index'),
    path('login/', views.login_view, name='login'),
    path('supervisor/', views.super_view, name='super'),
    path('pegawai/', views.pegawai_view, name='pegawai'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('add_employee/', add_employee, name='add_user'),
    ###
    #path('login', views.login, name="login"),
    path('add_employee/', add_employee, name='add_user'),
    path('train_data', views.train_data, name='train_data'),
    path('train_model/', train_model_view, name='train_model'),
    #path('delete_employee/<int:employee_id>/', views.delete_employee, name='delete_employee'),
    #path('logout/', views.logout, name='logout'),
    path('check_auth/', views.check_auth, name='check_auth'),
    path('save_location/', views.save_location, name='save_location'),
    path('log_login/', views.log_view, name='log'),
    path('divisi/', views.divisi, name='divisi'),
    

]

