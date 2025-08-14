from django.db import models

# Create your models here.                       
class Doctor(models.Model):
    name = models.CharField(max_length=100)
    speciality = models.CharField(max_length=50)
    is_available = models.BooleanField(default=True)
    image = models.ImageField(upload_to='doctors/', blank=True)

class Patient(models.Model):
    name = models.CharField(max_length=100)
    patient_id = models.CharField(max_length=20, unique=True)
    gender = models.CharField(max_length=10)
    admission_date = models.DateTimeField(auto_now_add=True)

class Appointment(models.Model):
    STATUS_CHOICES = [
        ('completed', 'Completed'),
        ('canceled', 'Canceled'),
        ('pending', 'Pending'),
    ]
    name = models.CharField(max_length=100)
    appointment_id = models.CharField(max_length=20)
    date = models.DateField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)