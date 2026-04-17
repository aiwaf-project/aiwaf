from django.urls import re_path
from .views import proxy


urlpatterns = [
    re_path(r"^(?P<path>.*)$", proxy),
]

