from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('start_camera/', views.start_camera, name='start_camera'),
    path('welcome/', views.bienvenido, name='welcome'),
]
