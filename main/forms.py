from django import forms
from .models import Employee
from django.forms.widgets import ClearableFileInput
from django.core.exceptions import ValidationError

class MultipleFileInput(ClearableFileInput):
    allow_multiple_selected = True

class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result

class LoginForm(forms.Form):
    username = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control form-control-user', 
                                    'placeholder': 'Enter Username'}),
        max_length=50
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control form-control-user', 
                                        'placeholder': 'Password'}),
    )

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get('username')
        password = cleaned_data.get('password')

        if not username or not password:
            raise ValidationError('Both username and password are required')
        return cleaned_data

class EmployeeForm(forms.ModelForm):
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control form-control-user'})
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control form-control-user'})
    )

    class Meta:
        model = Employee
        fields = ['name', 'role', 'email', 'address', 'username', 'password', 
                 'satker', 'jabatan', 'birthday', 'nip', 'phone']
        widgets = {
            'birthday': forms.DateInput(attrs={'type': 'date'})
        }

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')

        if password and confirm_password and password != confirm_password:
            raise ValidationError('Passwords do not match')

        # Password complexity validation
        if password:
            if len(password) < 8:
                raise ValidationError('Password must be at least 8 characters long')
            if not any(char.isdigit() for char in password):
                raise ValidationError('Password must contain at least one number')
            if not any(char.isupper() for char in password):
                raise ValidationError('Password must contain at least one uppercase letter')

        return cleaned_data

class PhotoUploadForm(forms.Form):
    photos = MultipleFileField(label='Upload Photos')