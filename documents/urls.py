from django.urls import path

from .schema_views import openapi_schema, swagger_ui
from .views import ChatView

app_name = "documents"

urlpatterns = [
    path("chat/", ChatView.as_view(), name="chat"),
    path("docs/", swagger_ui, name="docs"),
]
