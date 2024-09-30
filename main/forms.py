from django import forms
from main.models import Query

class QueryForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for visible in self.visible_fields():
            visible.field.widget.attrs['class'] = 'form-control'
    class Meta:
        model = Query
        fields = ("name","age", "gender", "symptoms", "history", "family", "lifestyle", "tests", "other",)   # NOTE: the trailing comma is required


