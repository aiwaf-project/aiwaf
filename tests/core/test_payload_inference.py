from aiwaf.core.api_detection import detect_api_endpoint
from aiwaf.core.path_manifest import classify_route


def flask_form_view():
    from flask import request

    username = request.form.get("username")
    password = request.form["password"]
    return {"username": username, "password": password}


def fastapi_json_view(request):
    payload = request.get_json()
    email = payload.get("email")
    message = payload["message"]
    return {"email": email, "message": message}


def test_payload_fields_are_inferred_from_form_and_subscript_access():
    detection = detect_api_endpoint(
        flask_form_view,
        framework="flask",
        path="/login",
        methods=["POST"],
    )

    assert detection.payload_type == "form"
    assert detection.payload_fields == ["password", "username"]


def test_payload_fields_are_inferred_from_json_payload_aliases():
    detection = detect_api_endpoint(
        fastapi_json_view,
        framework="fastapi",
        path="/api/contact",
        methods=["POST"],
    )

    assert detection.payload_type == "json"
    assert detection.payload_fields == ["email", "message"]


def test_classification_preserves_payload_fields():
    classified = classify_route(
        "/api/contact",
        methods=["POST"],
        metadata={
            "response_type": "json",
            "payload_type": "json",
            "api_confidence": 0.8,
            "payload_fields": ["email", "message"],
        },
    )

    assert classified["payload_fields"] == ["email", "message"]
