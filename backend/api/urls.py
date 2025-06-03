from django.urls import path
from .views import predict_task_duration

urlpatterns = [
    path('predict/', predict_task_duration, name='predict_task_duration'),
]
