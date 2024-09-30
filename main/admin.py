from django.contrib import admin

# Register your models here.
#from django.contrib import admin
from .models import Specialist

@admin.register(Specialist)
#class SpecialistAdmin(admin.ModelAdmin):
#    list_display = ('name', 'specialty', 'location','contact', 'experience')
class SpecialistAdmin(admin.ModelAdmin):
    list_display = ('name', 'specialty', 'available_day', 'phone')
