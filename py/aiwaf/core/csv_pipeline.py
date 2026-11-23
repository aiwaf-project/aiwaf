"""Explicit Rust-owned CSV pipeline; no automatic fallback or legacy config mixing."""
from __future__ import annotations

import importlib
import json
import inspect
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from .sql_injection import inspect_wsgi_request, inspect_asgi_request
from .storage_ops import now_iso

# Monitored SQL findings are logged as the regular chain logs them.
PAYLOAD_LOGGER = logging.getLogger("aiwaf.payload")
LOGGER = logging.getLogger("aiwaf.native")
SCHEMA_LOGGER = logging.getLogger("aiwaf.route_schema")

# Adapter-level keys. The engine rejects unknown profile keys, so these are read here.
_ADAPTER_KEYS = ("path_rules", "manifest_path", "injection_risk_mode", "store_cache_seconds")


class _Lookups:
    """Recent store answers per key, for ``seconds``, keeping at most ``limit`` keys."""

    def __init__(self, seconds, limit=10000):
        self.seconds, self.limit = seconds, limit
        self.items = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key, load):
        if self.seconds <= 0:
            return bool(load())
        now = time.monotonic()
        with self.lock:
            hit = self.items.get(key)
            if hit is not None and hit[0] > now:
                self.items.move_to_end(key)
                return hit[1]
        value = bool(load())
        self.set(key, value, now)
        return value

    def set(self, key, value, now=None):
        if self.seconds <= 0:
            return
        with self.lock:
            self.items[key] = ((time.monotonic() if now is None else now) + self.seconds, value)
            self.items.move_to_end(key)
            while len(self.items) > self.limit:
                self.items.popitem(last=False)


class PipelineStores:
    """The stores a native pipeline reads and writes instead of its own CSV files.

    Subclass it, or pass any object with these methods. Reads happen before each decision and
    writes after it, so the decision itself stays in aiwaf-rust.
    """

    def is_blocked(self, ip: str) -> bool:
        raise NotImplementedError

    def is_exempted(self, ip: str, path: str) -> bool:
        raise NotImplementedError

    def top_keywords(self, n: int) -> list:
        raise NotImplementedError

    def blocked_countries(self) -> list:
        raise NotImplementedError

    def block(self, ip: str, reason: str) -> None:
        raise NotImplementedError

    def learn(self, keywords: list) -> None:
        raise NotImplementedError


_STORE_METHODS = ("is_blocked", "is_exempted", "top_keywords", "blocked_countries", "block", "learn")
_RATE_OVERRIDE_KEYS = (("WINDOW", "rate_window"), ("MAX", "rate_max"), ("FLOOD", "rate_flood"))

STAGES = frozenset({"geo_block", "ip_keyword_block", "rate_limit", "ai_anomaly",
                    "honeypot", "uuid_tamper", "header_validation", "logging"})


@dataclass(frozen=True)
class _BufferPolicy:
    """Only the transport buffer contract; it performs no Python SQL decisions."""
    mode: str
    max_bytes: int

    @staticmethod
    def enforce(finding):
        return finding


def validate_fact_callback(callback):
    if callback is not None and (not callable(callback) or inspect.iscoroutinefunction(callback)):
        raise TypeError("Pipeline fact callbacks must be synchronous callables")


def fact_value(callback, value):
    result = callback(value)
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("Pipeline fact callbacks must return synchronous values")
    return result


class CSVPipeline:
    """Immutable profile plus Rust-owned state. Install instead of the legacy chain.

    ``config`` uses the native profile's lowercase keys; it does not implicitly
    consume framework settings, custom stores or legacy middleware callbacks.
    A missing/outdated native extension is a startup error, never a Python fallback.
    """
    def __init__(self, config=None, *, model=None, honeypot_cache=None, stores=None, rate_counter=None):
        """``stores`` replaces the native CSV blocklist, exemptions, keywords and countries.

        ``rate_counter`` (or ``rate_backend: "redis"``) shares rate counts across workers.
        """
        config = dict(config or {})
        self._configure_routes(config)
        self.stores = stores
        self._plans, self._plans_lock = OrderedDict(), threading.Lock()
        self._store_cache_seconds = float(config.pop("store_cache_seconds", 5))
        self._store_cache = {}
        # Blocklist and exemption answers, per IP and per IP and path, for the same few seconds.
        self._blocked = _Lookups(self._store_cache_seconds)
        self._exempted = _Lookups(self._store_cache_seconds)
        if stores is not None:
            missing = [name for name in _STORE_METHODS if not callable(getattr(stores, name, None))]
            if missing:
                raise TypeError(f"stores is missing: {', '.join(missing)}")
            config["policy_store"] = "host"
        backend = config.get("honeypot_backend", "csv")
        if backend in {"cache", "redis"}:
            from .honeypot_cache import HoneypotCache, RedisHoneypotCache
            namespace = config.pop("honeypot_cache_namespace", "aiwaf:native:honeypot:v1")
            if backend == "redis":
                if honeypot_cache is not None:
                    raise ValueError("Choose honeypot_backend=cache to supply a cache object")
                honeypot_cache = RedisHoneypotCache(config.pop("honeypot_redis_url", None))
            if honeypot_cache is None:
                raise ValueError("honeypot_backend=cache requires a honeypot_cache object")
            honeypot_cache = HoneypotCache(honeypot_cache, namespace=namespace)
            config["honeypot_backend"] = "cache"
        elif honeypot_cache is not None and backend != "django_cache":
            raise ValueError("A honeypot cache object requires honeypot_backend=cache")
        rate_counter = self._configure_rate_counter(config, rate_counter)
        self.profile = config
        self.needs_country = "geo_block" in config.get("enabled", STAGES)
        try:
            factory = importlib.import_module("aiwaf_rust").NativeCSVPipeline
        except (ImportError, AttributeError) as error:
            raise RuntimeError("The CSV pipeline requires aiwaf-rust with NativeCSVPipeline support") from error
        # Before construction: an older engine rejects newer profile values with a vaguer error.
        self._check_engine_supports(factory)
        supports = getattr(factory, "supports", None)
        self._uuid_fact = bool(supports and supports("uuid_fact"))
        self._uuid_fact_warned = False
        directory = config.get('data_dir', 'aiwaf_data')
        if isinstance(directory, str) and directory:
            from .policy_csv_storage import PolicyCSVStorage
            PolicyCSVStorage(data_dir=directory)
        wire_config = dict(config)
        self._configure_schema(wire_config, supports)
        if 'soft_block_blocklist' in wire_config:
            wire_config['soft_block_blacklist'] = wire_config.pop('soft_block_blocklist')
        profile_json = json.dumps(wire_config, allow_nan=False)
        if rate_counter is not None:
            self.native = factory(profile_json, honeypot_cache, rate_counter)
        elif honeypot_cache is None:
            self.native = factory(profile_json)
        else:
            try:
                self.native = factory(profile_json, honeypot_cache)
            except TypeError as error:
                raise RuntimeError("Cache honeypot timing requires aiwaf-rust>=0.2.7 with cache delegation support") from error
        self.policy = _BufferPolicy(config.get("sql_mode", "block"), config.get("max_bytes", 65536))
        if model is not None:
            self.native.set_model(model)

    def _configure_routes(self, config):
        """Read path rules, the route manifest and the route-check mode, as the regular chain does."""
        from .path_manifest import DEFAULT_MANIFEST_PATH, get_effective_path_rules
        from .route_risks import observer_for

        explicit = config.pop("path_rules", None) or []
        if not isinstance(explicit, (list, tuple)):
            raise ValueError("path_rules must be a list of rules")
        self.manifest_path = str(config.pop("manifest_path", None) or DEFAULT_MANIFEST_PATH)
        self.injection_risk_mode = config.pop("injection_risk_mode", None)
        # Validates the mode now, so a typo fails at startup rather than on a request.
        observer_for(self.injection_risk_mode, self.manifest_path, "/", schema_mode="off")
        self.path_rules = get_effective_path_rules(list(explicit), manifest_path=self.manifest_path)
        # The regular rate limiter flood-blocks above the flood threshold before a higher limit is
        # reached, so a limit above it acts as the threshold: the engine is given that.
        if "rate_max" in config and config.get("rate_flood", 40) < config["rate_max"]:
            config["rate_max"] = config.get("rate_flood", 40)
        self._rate_defaults = {"rate_window": config.get("rate_window", 10), "rate_max": config.get("rate_max", 20),
                               "rate_flood": config.get("rate_flood", 40)}
        for rule in self.path_rules:
            limits = self._effective_rate(_rate_overrides(rule.get("RATE_LIMIT", rule.get("rate_limit"))
                                                          if isinstance(rule, dict) else None))
            if limits and (limits["rate_window"] < 1 or limits["rate_max"] < 0):
                raise ValueError(f"Invalid RATE_LIMIT for path rule {rule.get('PREFIX')!r}")

    def _effective_rate(self, overrides):
        """A path rule's limits over the profile's, with a limit above the flood threshold acting as it."""
        if not overrides:
            return {}
        limits = {**self._rate_defaults, **overrides}
        limits["rate_max"] = min(limits["rate_max"], limits["rate_flood"])
        return limits

    @staticmethod
    def _configure_rate_counter(config, rate_counter):
        """The shared rate counter, from ``rate_backend: "redis"`` or a counter object."""
        backend = config.get("rate_backend", "memory")
        if backend == "redis":
            if rate_counter is not None:
                raise ValueError("Choose rate_backend=counter to supply a rate_counter object")
            from .rate_counter import RedisRateCounter
            rate_counter = RedisRateCounter(config.pop("rate_redis_url", None),
                                            namespace=config.pop("rate_cache_namespace", "aiwaf:native:rate:v1"))
            config["rate_backend"] = "counter"
        elif backend == "counter":
            if rate_counter is None:
                raise ValueError("rate_backend=counter requires a rate_counter object")
        elif rate_counter is not None:
            raise ValueError("A rate_counter object requires rate_backend=counter")
        if rate_counter is not None and (not callable(getattr(rate_counter, "count", None))
                                         or inspect.iscoroutinefunction(rate_counter.count)):
            raise TypeError("rate_counter requires a synchronous count(key, now, window_seconds) method")
        return rate_counter

    def _configure_schema(self, wire_config, supports):
        """Give the engine the manifest's saved inputs, as the regular chain reviews requests with.

        AIWAF_ROUTE_SCHEMA_MODE is monitor by default. An engine too old to review leaves it off.
        Strict inputs (AIWAF_STRICT_INPUTS) ride in that manifest, when the engine has them.
        """
        from .path_manifest import load_manifest
        from .route_risks import SCHEMA_MODES, SCHEMA_SETTING, STRICT_EXEMPT_WARNING, STRICT_WARNING, strict_inputs

        strict = strict_inputs(wire_config.pop("strict_inputs", None), wire_config.pop("strict_inputs_allow", None),
                               wire_config.pop("strict_inputs_exempt", None))
        mode = str(wire_config.pop("route_schema_mode", None) or os.getenv(SCHEMA_SETTING, "monitor")).strip().lower()
        if mode not in SCHEMA_MODES:
            raise ValueError("AIWAF_ROUTE_SCHEMA_MODE must be enforce, monitor, or off")
        manifest = load_manifest(self.manifest_path) if mode != "off" else None
        routes = manifest.get("routes") if isinstance(manifest, dict) else None
        if not isinstance(routes, dict) or not any(isinstance(entry, dict) and "inputs" in entry for entry in routes.values()):
            return
        if not (supports and supports("route_schema")):
            LOGGER.warning("Route schemas need a newer aiwaf-rust in the native pipeline; they're not reviewed. "
                           "Run: pip install -U aiwaf-rust")
            return
        wire_config["route_schema_mode"] = mode
        wire_config["route_manifest"] = manifest
        if strict and strict.get("exempt") and not supports("strict_exempt"):
            # An older engine would check the exempt paths too: better none at all.
            LOGGER.warning(STRICT_EXEMPT_WARNING)
        elif strict and supports("strict_inputs"):
            wire_config["route_manifest"] = {**manifest, "strict_inputs": strict}
        elif strict:
            LOGGER.warning(STRICT_WARNING)

    def _check_engine_supports(self, factory):
        """Refuse to start when the configuration needs a fact this aiwaf-rust build lacks."""
        from .route_risks import observer_for

        supports = getattr(factory, "supports", None)
        needed = []
        if any(isinstance(rule, dict) and _rate_overrides(rule.get("RATE_LIMIT", rule.get("rate_limit")))
               for rule in self.path_rules):
            needed.append("rate_overrides")
        if observer_for(self.injection_risk_mode, self.manifest_path, "/", schema_mode="off") is not None:
            needed.append("route_rule")
        if self.profile.get("sql_mode") == "monitor":
            needed.append("sql_monitor")
        if self.stores is not None:
            needed.append("host_store")
        if self.profile.get("rate_backend") == "counter":
            needed.append("rate_counter")
        missing = [name for name in needed if not (supports and supports(name))]
        if missing:
            raise RuntimeError(f"These native pipeline features need a newer aiwaf-rust: {', '.join(missing)}. "
                               "Run: pip install -U aiwaf-rust")

    def disabled_stages(self, path, endpoint=None):
        """The stages path rules and the view's decorators turn off for this path and view."""
        return set(self._planned(path, endpoint)[0])

    def _planned(self, path, endpoint):
        from .route_plan import get_route_execution_plan

        # Path rules and a view's exemption decorators are fixed once the app runs, so the plan for
        # a path and view is computed once. Route checks still inspect every request.
        try:
            key = (path, endpoint)
            hash(key)
        except TypeError:
            key = None
        with self._plans_lock:
            planned = self._plans.get(key) if key is not None else None
            if planned is not None:
                self._plans.move_to_end(key)
        if planned is None:
            required = set(getattr(endpoint, "_aiwaf_required_middlewares", ()) or ())
            full = bool(getattr(endpoint, "_aiwaf_exempt", False) or getattr(endpoint, "aiwaf_exempt", False))
            exempt = set(getattr(endpoint, "_aiwaf_exempt_middlewares", ()) or ())
            plan = get_route_execution_plan(path, self.path_rules, fully_exempt=full,
                                            exempt_middlewares=exempt, required_middlewares=required)
            planned = (frozenset(STAGES) - frozenset(plan.enabled_middlewares),
                       self._effective_rate(_rate_overrides(plan.get_rate_limit_overrides())))
            if key is not None:
                with self._plans_lock:
                    self._plans[key] = planned
                    while len(self._plans) > 4096:
                        self._plans.popitem(last=False)
        return planned

    def _route(self, path, method, endpoint, query, body):
        """Stages turned off for this request, its rate limits, and any route-check rule."""
        from .route_risks import observer_for

        planned = self._planned(path, endpoint)
        disabled, rate = planned
        rule = None
        if "ip_keyword_block" not in disabled and "ip_keyword_block" in self.profile.get("enabled", STAGES):
            observe = observer_for(self.injection_risk_mode, self.manifest_path, path, method, schema_mode="off")
            finding = observe(query, body) if observe else None
            rule = finding.rule if finding else None
        return set(disabled), dict(rate), rule

    def begin(self, *, ip, path, method, headers, query, body, exists, methods,
              authenticated=False, endpoint=None, country="", exhausted=False, uuid=None):
        """``uuid`` is a route's UUID parameter, as Django routes carry it; otherwise ``?uuid=`` is used."""
        disabled, rate, route_rule = self._route(path, method, endpoint, query or b"", body or b"")
        headers = dict(headers)
        headers["AIWAF_COUNTRY"] = country or ""
        now = time.time()
        ip = ip or "unknown"
        extra = {}
        exempt = False
        if self.stores is not None:
            # Only what this request's stages use: an exempt request runs logging alone.
            active = set(self.profile.get("enabled", STAGES)) - set(disabled)
            # A store that fails is logged and read as not exempt, not blocked and nothing learned,
            # so a storage outage doesn't turn every request into an error.
            if active - {"logging"}:
                exempt = self._read("exemptions", False, lambda: self._exempted.get(
                    (ip, path), lambda: self.stores.is_exempted(ip, path)))
            if exempt:
                active = set()
            if active & {"ip_keyword_block", "ai_anomaly"} and self.profile.get("ip_blocking", True):
                extra["blocked"] = self._read("blocklist", False, lambda: self._blocked.get(ip, lambda: self.stores.is_blocked(ip)))
            if "ip_keyword_block" in active and self.profile.get("learning", True):
                top_n = int(self.profile.get("top_n", 10))
                extra["dynamic_keywords"] = list(self._read("keywords", (), lambda: self._cached(
                    "keywords", lambda: self.stores.top_keywords(top_n))))
            if "geo_block" in active:
                extra["geo_countries"] = set(self._read("countries", (), lambda: self._cached(
                    "countries", self.stores.blocked_countries)))
        args = (ip, path, method, headers, query or b"", body or b"",
                bool(exists), "GET" in methods, "POST" in methods, method.upper() in methods,
                bool(authenticated), disabled, exempt, now, now_iso(), bool(exhausted))
        if uuid is not None:
            if self._uuid_fact:
                extra["uuid"] = str(uuid)
            elif not self._uuid_fact_warned:
                self._uuid_fact_warned = True
                LOGGER.warning("Route UUIDs need a newer aiwaf-rust to be checked; only ?uuid= is checked. "
                               "Run: pip install -U aiwaf-rust")
        if route_rule is None and not rate and not extra:
            # Older aiwaf-rust builds take only these arguments.
            result = self.native.begin(*args)
        else:
            result = self.native.begin(*args, route_rule, rate.get("rate_window"), rate.get("rate_max"),
                                       rate.get("rate_flood"), **extra)
        monitored = getattr(result[0], "sql_monitor", None)
        if monitored:
            PAYLOAD_LOGGER.warning("aiwaf.payload rule=%s mode=monitor", monitored)
        reviewed = getattr(result[0], "schema_event", None)
        if reviewed:
            SCHEMA_LOGGER.warning("%s", reviewed)
        self._apply_effects(result[0])
        return result

    def finish(self, context, status, content_length="-"):
        result = self.native.finish(context, int(status), time.time(), str(content_length), True)
        self._apply_effects(context)
        return result

    def _cached(self, name, load):
        """Keywords and countries change rarely, so they're read at most every store_cache_seconds.

        Blocklist and exemption answers are kept as long, per IP and path, so a block or exemption
        made elsewhere (another worker, the admin or CLI) can take that long to apply here.
        """
        now = time.monotonic()
        cached = self._store_cache.get(name)
        if cached is None or now >= cached[0]:
            cached = self._store_cache[name] = (now + self._store_cache_seconds, list(load() or ()))
        return cached[1]

    @staticmethod
    def _read(store, fallback, load):
        try:
            return load()
        except Exception as error:
            LOGGER.warning("AIWAF store read failed (%s): %s; continuing without it", store, error)
            return fallback

    def _apply_effects(self, context):
        """Write the engine's blocks and learned keywords to the host's stores."""
        if self.stores is None:
            return
        for kind, values in context.take_effects():
            try:
                if kind == "block":
                    # This worker's own blocks apply at once, whatever was cached or written.
                    self._blocked.set(values[0], True)
                    self.stores.block(values[0], values[1])
                elif kind == "keywords" and values:
                    self.stores.learn(list(values))
                    self._store_cache.pop("keywords", None)
            except Exception as error:
                # The decision stands; a failed write is logged rather than failing the response.
                LOGGER.warning("AIWAF store write failed (%s): %s", kind, error)

    def capture_wsgi(self, request, *, django=False):
        captured = [b"", None]

        def capture(query, body):
            captured[:] = [query, body]
            return None

        finding = inspect_wsgi_request(request, self.policy, django=django, evaluator=capture)
        query = captured[0]
        if isinstance(query, str):
            query = query.encode("utf-8", errors="replace")
        return query, captured[1], bool(finding)

    async def capture_asgi(self, request):
        captured = [request.scope.get("query_string", b""), None]

        def capture(query, body):
            captured[:] = [query, body]
            return None

        finding = await inspect_asgi_request(request, self.policy, evaluator=capture)
        return captured[0], captured[1], bool(finding)


def _rate_overrides(overrides):
    """WINDOW, MAX and FLOOD from a path rule's RATE_LIMIT, as numbers for the engine."""
    result = {}
    for key, name in _RATE_OVERRIDE_KEYS:
        value = (overrides or {}).get(key, (overrides or {}).get(key.lower()))
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"RATE_LIMIT {key} must be a number")
        result[name] = float(value) if name == "rate_window" else int(value)
    return result


def environ_headers(headers, protocol="HTTP/1.1"):
    result = {("CONTENT_TYPE" if k.lower() == "content-type" else
               "CONTENT_LENGTH" if k.lower() == "content-length" else
               "HTTP_" + k.upper().replace("-", "_")): str(v) for k, v in headers.items()}
    result["SERVER_PROTOCOL"] = protocol
    return result
