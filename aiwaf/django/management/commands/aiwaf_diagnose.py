from django.core.management.base import BaseCommand
from django.conf import settings
import sys

class Command(BaseCommand):
    help = 'Diagnose AI-WAF installation and middleware setup'

    def handle(self, *args, **options):
        self.stdout.write(self.style.HTTP_INFO(" AI-WAF Installation Diagnostics"))
        self.stdout.write("")
        
        # Check AI-WAF import
        try:
            import aiwaf
            version = getattr(aiwaf, '__version__', 'Unknown')
            self.stdout.write(self.style.SUCCESS(f" AI-WAF imported successfully (version: {version})"))
            
            # Test critical imports that caused AppRegistryNotReady
            try:
                from aiwaf.django import storage, utils, trainer
                self.stdout.write(self.style.SUCCESS(" Critical modules (storage, utils, trainer) imported successfully"))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f" Critical module import failed: {e}"))
                
        except ImportError as e:
            self.stdout.write(self.style.ERROR(f" AI-WAF import failed: {e}"))
            return
        
        # Check if aiwaf.django is in INSTALLED_APPS
        installed_apps = getattr(settings, 'INSTALLED_APPS', [])
        if 'aiwaf.django' in installed_apps:
            self.stdout.write(self.style.SUCCESS(" 'aiwaf.django' found in INSTALLED_APPS"))
        else:
            self.stdout.write(self.style.ERROR(" 'aiwaf.django' NOT found in INSTALLED_APPS"))
            self.stdout.write(self.style.WARNING("   Add 'aiwaf.django' to your INSTALLED_APPS in settings.py"))
        
        # Check middleware availability
        self.stdout.write("")
        self.stdout.write(self.style.HTTP_INFO(" Middleware Availability Check:"))
        
        middleware_classes = [
            'IPAndKeywordBlockMiddleware',
            'RateLimitMiddleware', 
            'AIAnomalyMiddleware',
            'HoneypotTimingMiddleware',
            'UUIDTamperMiddleware',
            'GeoBlockMiddleware',
        ]
        
        try:
            import aiwaf.django.middleware as mw
            available_classes = dir(mw)
            
            for middleware_class in middleware_classes:
                if middleware_class in available_classes:
                    self.stdout.write(self.style.SUCCESS(f"   {middleware_class}"))
                else:
                    self.stdout.write(self.style.ERROR(f"   {middleware_class} (missing)"))
        except ImportError as e:
            self.stdout.write(self.style.ERROR(f" Could not import aiwaf.django.middleware: {e}"))
        
        # Check configured middleware
        self.stdout.write("")
        self.stdout.write(self.style.HTTP_INFO(" Configured Middleware:"))
        
        middleware_setting = getattr(settings, 'MIDDLEWARE', [])
        aiwaf_middleware = [mw for mw in middleware_setting if 'aiwaf' in mw.lower()]
        
        if aiwaf_middleware:
            for mw in aiwaf_middleware:
                # Test if middleware can be imported
                try:
                    from django.utils.module_loading import import_string
                    import_string(mw)
                    self.stdout.write(self.style.SUCCESS(f"   {mw}"))
                except ImportError as e:
                    self.stdout.write(self.style.ERROR(f"   {mw} - Import Error: {e}"))
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f"    {mw} - Warning: {e}"))
        else:
            self.stdout.write(self.style.WARNING("  No AI-WAF middleware found in MIDDLEWARE setting"))
        
        # Check storage configuration
        self.stdout.write("")
        self.stdout.write(self.style.HTTP_INFO(" Storage Configuration:"))
        
        storage_mode = getattr(settings, 'AIWAF_STORAGE_MODE', 'models')
        self.stdout.write(f"  Storage Mode: {storage_mode}")
        
        if storage_mode == 'csv':
            csv_dir = getattr(settings, 'AIWAF_CSV_DATA_DIR', 'aiwaf_data')
            self.stdout.write(f"  CSV Directory: {csv_dir}")
        
        # Check access log
        access_log = getattr(settings, 'AIWAF_ACCESS_LOG', None)
        if access_log:
            import os
            if os.path.exists(access_log):
                self.stdout.write(self.style.SUCCESS(f"   Access log found: {access_log}"))
            else:
                self.stdout.write(self.style.WARNING(f"    Access log not found: {access_log}"))
        else:
            self.stdout.write(self.style.WARNING("    AIWAF_ACCESS_LOG not configured"))
        
        # Check middleware logging
        middleware_logging = getattr(settings, 'AIWAF_MIDDLEWARE_LOGGING', False)
        if middleware_logging:
            self.stdout.write(self.style.SUCCESS("   Middleware logging enabled"))
        else:
            self.stdout.write(self.style.WARNING("    Middleware logging disabled"))

        # Check geo-blocking configuration
        self.stdout.write("")
        self.stdout.write(self.style.HTTP_INFO(" Geo-Blocking Check:"))
        geo_enabled = getattr(settings, 'AIWAF_GEO_BLOCK_ENABLED', False)
        if geo_enabled:
            self.stdout.write(self.style.SUCCESS("   Geo-blocking enabled"))
        else:
            self.stdout.write(self.style.WARNING("    Geo-blocking disabled"))

        geo_db_path = getattr(settings, 'AIWAF_GEOIP_DB_PATH', None)
        if geo_db_path:
            import os
            if os.path.exists(geo_db_path):
                self.stdout.write(self.style.SUCCESS(f"   GeoIP DB found: {geo_db_path}"))
            else:
                self.stdout.write(self.style.WARNING(f"    GeoIP DB not found: {geo_db_path}"))
        else:
            self.stdout.write(self.style.WARNING("    AIWAF_GEOIP_DB_PATH not configured"))
        
        # Recommendations
        self.stdout.write("")
        self.stdout.write(self.style.HTTP_INFO(" Recommendations:"))
        
        if 'aiwaf' not in installed_apps:
            self.stdout.write("  1. Add 'aiwaf' to INSTALLED_APPS")
            
        if not aiwaf_middleware:
            self.stdout.write("  2. Add AI-WAF middleware to MIDDLEWARE setting")
            
        if not access_log and not middleware_logging:
            self.stdout.write("  3. Configure AIWAF_ACCESS_LOG or enable AIWAF_MIDDLEWARE_LOGGING")
        
        # Quick fix commands
        self.stdout.write("")
        self.stdout.write(self.style.HTTP_INFO(" Quick Fix Commands:"))
        self.stdout.write("")
        self.stdout.write("# Update AI-WAF to latest version:")
        self.stdout.write("pip install --upgrade aiwaf")
        self.stdout.write("")
        self.stdout.write("# Run migrations (if using models mode):")
        self.stdout.write("python manage.py migrate")
        self.stdout.write("")
        self.stdout.write("# Test AI-WAF commands:")
        self.stdout.write("python manage.py add_ipexemption 127.0.0.1 --reason Testing")
