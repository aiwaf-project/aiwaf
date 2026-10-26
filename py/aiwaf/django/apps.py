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
        from pathlib import Path
        from django.conf import settings
        from aiwaf.core.edge_bridge import edge_configuration
        path = getattr(settings, 'AIWAF_BOUVANE_CONFIG', Path(getattr(settings, 'BASE_DIR', '.')) / '.bouvane.json')
        middleware = 'aiwaf.django.edge.EdgeAssistedMiddleware'
        if edge_configuration(path) and middleware not in settings.MIDDLEWARE:
            settings.MIDDLEWARE = [middleware, *settings.MIDDLEWARE]
