from django.urls import path
from main import views
from main.models import Query

urlpatterns = [
    path("", views.home, name="home"),
    path('recommendation/', views.recommendation, name='recommendation'),
   # path('appointment/<int:doctor_id>/', views.appointment_page, name='appointment_page'),
]