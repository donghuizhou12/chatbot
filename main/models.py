from django.db import models
from django.utils import timezone

class Query(models.Model):
    name = models.CharField(null=True, max_length=50)
    age = models.PositiveSmallIntegerField(null=True)
    gender = models.CharField(null=True, max_length=20)
    symptoms = models.CharField(null=True, max_length=500)
    history = models.CharField(null=True, max_length=500)
    family = models.CharField(null=True, max_length=500)
    lifestyle = models.CharField(null=True, max_length=500)
    tests = models.CharField(null=True, max_length=500)
    other = models.CharField(null=True, max_length=500)
    log_date = models.DateTimeField("date logged")

    def __str__(self):
        """Returns a string representation of a query."""
        date = timezone.localtime(self.log_date)
        return f"'{self.name}''{self.age}' '{self.gender}' '{self.symptoms}' '{self.history}' '{self.family}' '{self.lifestyle}' '{self.tests}' '{self.other}'  logged on {date.strftime('%A, %d %B, %Y at %X')}"

#from django.db import models

#class Specialist(models.Model):
 #   name = models.CharField(max_length=255)
  #  specialty = models.CharField(max_length=255)
   # contact = models.CharField(max_length=255)

    #def __str__(self):
     #   return self.name

#class Specialist(models.Model):
 #   name = models.CharField(max_length=255)
 #   specialty = models.CharField(max_length=255)
 #   location = models.CharField(max_length=255, null=True, blank=True)
 #   contact = models.CharField(max_length=255, null=True, blank=True)
 #   experience = models.IntegerField(null=True, blank=True)

  #  def __str__(self):
 #       return self.name

class Specialist(models.Model):
    name = models.CharField(max_length=100)
    specialty = models.CharField(max_length=100)
    available_day = models.CharField(max_length=100, default="Monday")  # New field to replace location
    phone = models.CharField(max_length=15, default="000-000-0000")



