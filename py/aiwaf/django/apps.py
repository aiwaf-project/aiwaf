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
        # AIWAF_RECORD_INPUTS: record what each view reads, for init, with no MIDDLEWARE line.
        from aiwaf.core.input_recorder import add_django_recorder
        add_django_recorder()
