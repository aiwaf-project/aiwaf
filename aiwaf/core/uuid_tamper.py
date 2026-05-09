"""Shared UUID tamper helpers used by framework adapters."""

import re


UUID_RE = re.compile(r"^[a-f0-9\-]{36}$")


def is_malformed_uuid(value):
    """Return True when a UUID-like input is present but fails format validation."""
    if not value:
        return False
    return UUID_RE.match(str(value)) is None


def collect_uuid_model_fields(models, uuid_field_class):
    """
    Collect UUID lookup candidates from models.

    Returns tuples of ``(Model, field_name)`` for UUID primary keys and unique UUID fields.
    """
    uuid_fields = []
    for model in models:
        pk_field = model._meta.pk
        if isinstance(pk_field, uuid_field_class):
            uuid_fields.append((model, "pk"))
        for field in model._meta.fields:
            if field is pk_field:
                continue
            if isinstance(field, uuid_field_class) and getattr(field, "unique", False):
                uuid_fields.append((model, field.name))
    return uuid_fields


def uuid_exists_in_model_fields(uid, uuid_fields):
    """Return True if ``uid`` exists in any configured model UUID field candidate."""
    for model, field_name in uuid_fields:
        try:
            if field_name == "pk":
                if model.objects.filter(pk=uid).exists():
                    return True
            else:
                if model.objects.filter(**{field_name: uid}).exists():
                    return True
        except (ValueError, TypeError):
            continue
    return False
