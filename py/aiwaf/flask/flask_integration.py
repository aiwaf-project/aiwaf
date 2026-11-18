from __future__ import annotations

from typing import Iterable

from .ip_and_keyword_block_middleware import IPAndKeywordBlockMiddleware
from .rate_limit_middleware import RateLimitMiddleware
from .honeypot_timing_middleware import HoneypotTimingMiddleware
from .header_validation_middleware import HeaderValidationMiddleware
from .anomaly_middleware import AIAnomalyMiddleware
from .uuid_tamper_middleware import UUIDTamperMiddleware
from .logging_middleware import AIWAFLoggingMiddleware
from .geo_block_middleware import GeoBlockMiddleware
from aiwaf.core.middleware_plan import plan_enabled_middlewares
from aiwaf.core.middleware_plan import should_enable_geo
from aiwaf.core.route_capabilities import detect_uuid_routes_in_flask_app
from .storage import get_geo_blocked_countries


class AIWAF:
    AVAILABLE_MIDDLEWARES = {
        # Keep registration order aligned with Django chain semantics.
        "geo_block": GeoBlockMiddleware,
        "ip_keyword_block": IPAndKeywordBlockMiddleware,
        "rate_limit": RateLimitMiddleware,
        "ai_anomaly": AIAnomalyMiddleware,
        "honeypot": HoneypotTimingMiddleware,
        "uuid_tamper": UUIDTamperMiddleware,
        "header_validation": HeaderValidationMiddleware,
        "logging": AIWAFLoggingMiddleware,
    }

    def __init__(self, app=None, middlewares: Iterable[str] | None = None, disable_middlewares: Iterable[str] | None = None, use_database=None,
                 native_pipeline=None):
        """``native_pipeline=False`` keeps the regular middleware when the native one would be the default."""
        self.app = None
        self.middleware_instances = {}
        self.enabled_middlewares = set()
        self.native = None
        self.native_pipeline = native_pipeline
        if app is not None:
            self.init_app(app, middlewares=middlewares, disable_middlewares=disable_middlewares, use_database=use_database)

    @classmethod
    def list_available_middlewares(cls):
        return list(cls.AVAILABLE_MIDDLEWARES.keys())

    def get_enabled_middlewares(self):
        return list(self.enabled_middlewares)

    def is_middleware_enabled(self, name: str) -> bool:
        return name in self.enabled_middlewares

    def _resolve_middlewares(self, middlewares, disable_middlewares):
        all_keys = list(self.AVAILABLE_MIDDLEWARES.keys())
        access_log = self.app.config.get("AIWAF_ACCESS_LOG") if self.app is not None else None
        geo_enabled_flag = bool(self.app.config.get("AIWAF_GEO_BLOCK_ENABLED", False)) if self.app is not None else False
        static_block = self.app.config.get("AIWAF_GEO_BLOCK_COUNTRIES", []) if self.app is not None else []
        dynamic_block = get_geo_blocked_countries() if self.app is not None else []
        return plan_enabled_middlewares(
            ordered_available=all_keys,
            requested=middlewares,
            disabled=disable_middlewares,
            access_log=access_log,
            geo_enabled_flag=geo_enabled_flag,
            static_block_countries=static_block,
            dynamic_block_countries=dynamic_block,
            has_uuid_routes=detect_uuid_routes_in_flask_app(self.app),
        )

    def _native_default(self, app, enabled, use_database):
        """The native pipeline, when it can run these exact settings; otherwise None."""
        from aiwaf.core.native_settings import (carry_model, log_native_default, log_native_fallback,
                                                native_access_log, native_by_default, native_config)

        from aiwaf.core import profiles

        if use_database is True:
            return None  # A database chosen in code, which the converter can't see.
        if profiles.monitoring():
            return None  # Monitor runs the regular middleware, whose every block can be logged instead.
        names = {name: value for name, value in app.config.items() if name.startswith("AIWAF_")}
        conversion = native_config(names, "flask")
        # The same stages the regular middleware would register.
        stages = [stage for stage in self.AVAILABLE_MIDDLEWARES if stage in set(enabled)]
        # A log format the app chose (JSON, CSV or combined) is written by the regular logging
        # middleware, to the same files; otherwise the native pipeline logs requests itself.
        regular_log = "logging" in stages and app.config.get("AIWAF_LOG_FORMAT") is not None
        if regular_log:
            stages.remove("logging")
        elif "logging" in stages:
            native_access_log(conversion, "AIWAF_LOG_DIR", app.config.get("AIWAF_LOG_DIR", "aiwaf_logs"))
        conversion.config["enabled"] = stages
        if "ip_keyword_block" in stages or "ai_anomaly" in stages:
            # Words the regular keyword check never learns: the app's routes, AIWAF_ALLOWED_PATH_KEYWORDS
            # and AIWAF_EXEMPT_KEYWORDS, read as it reads them.
            from .ip_and_keyword_block_middleware import IPAndKeywordBlockMiddleware

            conversion.config["legitimate_keywords"] = sorted(
                IPAndKeywordBlockMiddleware._get_legitimate_path_keywords(None, app))
            # Keyword blocks skip paths under the app's own route prefixes, as in the regular check.
            conversion.config["safe_prefixes"] = sorted(IPAndKeywordBlockMiddleware._collect_safe_prefixes(None, app))
        if "ai_anomaly" in stages:
            from aiwaf.core.native_settings import carry_learning
            from .anomaly_middleware import STATIC_KEYWORDS

            carry_learning(conversion, "flask", STATIC_KEYWORDS)
        # The model settings are decided here, from the model the regular middleware would use.
        # AIWAF_LOG_DIR only places the request log, decided above, and AIWAF_ACCESS_LOG only
        # counts toward the stages and the model's logs, both decided from it already.
        handled = {"AIWAF_MODEL_PATH", "AIWAF_ALLOW_CUSTOM_MODEL_PATH", "AIWAF_MIN_AI_LOGS", "AIWAF_FORCE_AI",
                   "AIWAF_LOG_DIR", "AIWAF_ACCESS_LOG", "AIWAF_ALLOWED_PATH_KEYWORDS"} | ({"AIWAF_LOG_FORMAT", "AIWAF_CSV_LOG_FSYNC"} if regular_log else set())
        conversion.notes = [note for note in conversion.notes if note.setting not in handled]
        model = self._regular_model(app) if "ai_anomaly" in enabled else None
        if model is not None and not self._model_in_use(app):
            conversion.note("AIWAF_MIN_AI_LOGS", "changed", "The regular middleware holds the model back until there "
                                                            "are enough logs; the native pipeline would use it at once.")
        model = carry_model(conversion, model, "AIWAF_MODEL_PATH")
        choice = self.native_pipeline if self.native_pipeline is not None else app.config.get("AIWAF_NATIVE_PIPELINE")
        if not native_by_default(conversion, choice=choice, framework="Flask"):
            return None
        from .csv_pipeline import NativeAIWAF

        try:
            native = NativeAIWAF(app, config=dict(conversion.config), model=model)
        except Exception as error:
            log_native_fallback("Flask", error)
            return None
        if regular_log:
            # Registered after the native pipeline, as the regular chain registers it last.
            self.middleware_instances["logging"] = self.AVAILABLE_MIDDLEWARES["logging"](app)
        log_native_default("Flask")
        return native

    @staticmethod
    def _model_in_use(app):
        """Whether the regular anomaly middleware uses its model now: forced, or with enough logs."""
        import logging
        from types import SimpleNamespace
        from .anomaly_middleware import AIAnomalyMiddleware

        if app.config.get("AIWAF_FORCE_AI", False):
            return True
        return AIAnomalyMiddleware._check_log_data_sufficiency(SimpleNamespace(logger=logging.getLogger("aiwaf.native")), app)

    @staticmethod
    def _regular_model(app):
        """The model the regular anomaly middleware would load, if any, read the same way."""
        import os
        from aiwaf.core.model_security import is_trusted_model_path
        from aiwaf.core.model_serialization import load_model_artifact
        from aiwaf.core.rust_backend import rust_isolation_forest_from_json

        default = os.path.join(os.path.dirname(__file__), "resources", "model.json")
        path = app.config.get("AIWAF_MODEL_PATH", default)
        trusted = is_trusted_model_path(path, default_path=default,
                                        allow_custom=bool(app.config.get("AIWAF_ALLOW_CUSTOM_MODEL_PATH", False)))
        if not trusted or not os.path.exists(path):
            return None
        try:
            data = load_model_artifact(path)
        except Exception:
            return None  # The regular middleware runs without a model then too.
        if isinstance(data, dict) and data.get("model_type") == "aiwaf_rust.IsolationForest":
            return rust_isolation_forest_from_json(data.get("model_state")) or data
        return data.get("model", data) if isinstance(data, dict) else data

    def init_app(self, app, middlewares: Iterable[str] | None = None, disable_middlewares: Iterable[str] | None = None, use_database=None):
        self.app = app
        from aiwaf.core import profiles

        # AIWAF_PROFILE fills in what isn't set explicitly; Flask's rate defaults are 20 and 40.
        profiles.apply(app.config, app.config.setdefault, rate_max=20, rate_flood=40)
        if app.config.get('AIWAF_USE_CSV', True) and use_database is not True:
            from aiwaf.core.policy_csv_storage import PolicyCSVStorage
            from .storage import _get_data_dir
            with app.app_context():
                app.extensions['aiwaf_policy_storage'] = PolicyCSVStorage(data_dir=_get_data_dir())
        enabled = self._resolve_middlewares(middlewares, disable_middlewares)
        self.enabled_middlewares = set(enabled)
        if "geo_block" in enabled and not app.config.get("AIWAF_GEO_BLOCK_ENABLED", False):
            app.config["AIWAF_GEO_BLOCK_ENABLED"] = should_enable_geo(
                geo_enabled_flag=bool(app.config.get("AIWAF_GEO_BLOCK_ENABLED", False)),
                static_block_countries=app.config.get("AIWAF_GEO_BLOCK_COUNTRIES", []) or [],
                dynamic_block_countries=get_geo_blocked_countries(),
            )

        self.native = self._native_default(app, enabled, use_database)
        if self.native is not None:
            app.extensions = getattr(app, "extensions", {})
            app.extensions["aiwaf"] = self
            return self

        # ensure database initialized when requested
        if use_database is True:
            from .middleware import _init_database
            _init_database(app)

        for name in self.AVAILABLE_MIDDLEWARES:
            if name not in enabled:
                continue
            middleware_cls = self.AVAILABLE_MIDDLEWARES[name]
            instance = middleware_cls(app)
            self.middleware_instances[name] = instance

        app.extensions = getattr(app, "extensions", {})
        app.extensions["aiwaf"] = self
        return self
