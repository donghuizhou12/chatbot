from .models import Specialist

def find_specialist(symptom):
    specialist = Specialist.objects.filter(specialty=symptom.specialty, available_day='Monday')  # Adjust logic as needed
    return specialist