from django.apps import AppConfig

class AiwafConfig(AppConfig):
    name = "aiwaf.django"
    label = "aiwaf"
    verbose_name = "AIDriven Web Application Firewall"

    def ready(self):
        try:
            from .settings_compat import apply_legacy_settings
            apply_legacy_settings()
        except Exception:
            pass
