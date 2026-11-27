default_app_config = "aiwaf.django.apps.AiwafConfig"

__version__ = "1.3.4"

# Note: Middleware classes are available from aiwaf.django.middleware
# Import them only when needed to avoid circular imports during Django app loading


def __getattr__(name):
    # Loaded on first use, so importing aiwaf stays light while Django loads its apps.
    if name == "inputs":
        from aiwaf.core.input_sources import inputs

        return inputs
    raise AttributeError(f"module 'aiwaf' has no attribute {name!r}")
