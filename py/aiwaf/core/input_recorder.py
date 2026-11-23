"""Record which query and body names each view reads, while tests or staging traffic run.

Set ``AIWAF_RECORD_INPUTS`` to a file path (or to 1 for ``.aiwaf/observed_inputs.jsonl``) and add
the hook for your framework:

- Django: ``"aiwaf.core.input_recorder.DjangoInputRecorder"`` last in ``MIDDLEWARE``.
- Flask: ``record_flask_inputs(app)``.
- FastAPI and Starlette: ``app.add_middleware(AsgiInputRecorder)``.

Each request appends a line per name the view read: ``{"route", "method", "key"}``, with key
``*`` when it read a container whole (iterated it, or turned it into a dict). Names are what the
code read, not what the client sent, so a client's extra parameters aren't recorded. ``init``
reads the file (see input_sources): recorded names are added to the routes' inputs, and one the
analysis missed makes its route incomplete.

Without ``AIWAF_RECORD_INPUTS`` the hooks do nothing and nothing is patched. Reads of the raw
body, as ``json.loads(request.body)`` in Django, can't be seen.
"""

from __future__ import annotations

import contextvars
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

ENVIRONMENT = "AIWAF_RECORD_INPUTS"
DEFAULT_PATH = ".aiwaf/observed_inputs.jsonl"
WHOLE = "*"
# Read by framework middleware on every form post, not by the view.
_IGNORED = {"csrfmiddlewaretoken", "csrf_token"}

_KEYED = ("__getitem__", "get", "getlist", "getall", "pop")
_WHOLE_READS = ("items", "keys", "values", "lists", "dict", "to_dict", "copy", "urlencode", "multi_items", "__iter__")

_current: contextvars.ContextVar[Optional["Recording"]] = contextvars.ContextVar("aiwaf_input_recording", default=None)
_installed: Set[str] = set()


def destination(value: Optional[str] = None) -> Optional[Path]:
    """Where to record, from AIWAF_RECORD_INPUTS, or None when recording is off."""
    raw = (os.environ.get(ENVIRONMENT, "") if value is None else value).strip()
    if not raw or raw.lower() in {"0", "false", "no", "off"}:
        return None
    if raw.lower() in {"1", "true", "yes", "on"}:
        return Path(DEFAULT_PATH)
    return Path(raw)


class Recording:
    """The names one request's view read. owns says whether a container is this request's."""

    def __init__(self, owns: Callable[[Any], bool] = lambda container: False, scope: Any = None):
        self._owns = owns
        self.scope = scope
        self.owned: Set[int] = set()
        self.keys: Set[str] = set()
        self.whole: Set[int] = set()
        # Inside a read already noted, as items() calling __getitem__ for each key.
        self.busy = False

    def own(self, container: Any) -> Any:
        self.owned.add(id(container))
        return container

    def mine(self, container: Any) -> bool:
        if id(container) in self.owned:
            return True
        try:
            return bool(self._owns(container))
        except Exception:  # noqa: BLE001 - an owner check must never break the request
            return False

    def note(self, container: Any, key: Any) -> None:
        if id(container) in self.whole or not self.mine(container):
            return
        if key is WHOLE:
            # Read whole, so its route is incomplete anyway; the reads a lazy items() makes
            # afterwards aren't the view's own.
            self.whole.add(id(container))
            self.keys.add(WHOLE)
        elif str(key) not in _IGNORED:
            self.keys.add(str(key))

    def wrap(self, data: Any) -> Any:
        """A parsed JSON body, as a dict that records reads; anything else as it is."""
        return self.own(_RecordedDict(data)) if type(data) is dict else data

    def save(self, route: Optional[str], method: str, path: Optional[Path]) -> None:
        if not route or not path or not self.keys:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = "".join(json.dumps({"route": route, "method": method.upper(), "key": key}) + "\n" for key in sorted(self.keys))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(lines)


def current() -> Optional[Recording]:
    return _current.get()


class _RecordedDict(dict):
    """A JSON body that notes which keys the view reads."""

    def __getitem__(self, key):
        recording = _current.get()
        if recording is not None:
            recording.note(self, key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        recording = _current.get()
        if recording is not None:
            recording.note(self, key)
        return super().get(key, default)

    def __iter__(self):
        recording = _current.get()
        if recording is not None:
            recording.note(self, WHOLE)
        return super().__iter__()

    def items(self):
        recording = _current.get()
        if recording is not None:
            recording.note(self, WHOLE)
        return super().items()


def _patch(cls: type, label: str) -> None:
    """Wraps a container class's reads so they're noted for the request that owns the container."""
    if label in _installed:
        return
    _installed.add(label)
    for name in (*_KEYED, *_WHOLE_READS):
        # Inherited ones too, set on this class only, as QueryDict's get from MultiValueDict.
        original = getattr(cls, name, None)
        if original is None or not callable(original):
            continue
        keyed = name in _KEYED

        def wrapper(self, *args, __original=original, __keyed=keyed, **kwargs):
            recording = _current.get()
            if recording is None or recording.busy:
                return __original(self, *args, **kwargs)
            recording.note(self, args[0] if __keyed and args else WHOLE)
            recording.busy = True
            try:
                return __original(self, *args, **kwargs)
            finally:
                recording.busy = False

        wrapper.__name__ = name
        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        setattr(cls, name, wrapper)


def install() -> None:
    """Patch the containers of every framework that's installed. Only the hooks call this."""
    try:
        from django.http import QueryDict

        _patch(QueryDict, "django")
    except ImportError:
        pass
    try:
        from werkzeug.datastructures import MultiDict
        from werkzeug.wrappers import Request as WerkzeugRequest

        _patch(MultiDict, "werkzeug")
        if "werkzeug.json" not in _installed:
            _installed.add("werkzeug.json")
            werkzeug_get_json = WerkzeugRequest.get_json

            def recorded_get_json(self, *args, **kwargs):
                data = werkzeug_get_json(self, *args, **kwargs)
                recording = _current.get()
                return recording.wrap(data) if recording is not None and recording.mine(self) else data

            WerkzeugRequest.get_json = recorded_get_json  # type: ignore[method-assign]
    except ImportError:
        pass
    try:
        from starlette.datastructures import ImmutableMultiDict
        from starlette.requests import HTTPConnection, Request as StarletteRequest

        _patch(ImmutableMultiDict, "starlette")
        if "starlette.request" not in _installed:
            _installed.add("starlette.request")
            query_params = HTTPConnection.query_params
            get_form = StarletteRequest._get_form
            starlette_json = StarletteRequest.json

            def recorded_query_params(self):
                params = query_params.fget(self)
                recording = _current.get()
                if recording is not None and recording.scope is self.scope:
                    recording.own(params)
                return params

            async def recorded_get_form(self, *args, **kwargs):
                form = await get_form(self, *args, **kwargs)
                recording = _current.get()
                if recording is not None and recording.scope is self.scope:
                    recording.own(form)
                return form

            async def recorded_json(self):
                data = await starlette_json(self)
                recording = _current.get()
                return recording.wrap(data) if recording is not None and recording.scope is self.scope else data

            HTTPConnection.query_params = property(recorded_query_params)  # type: ignore[assignment]
            StarletteRequest._get_form = recorded_get_form  # type: ignore[method-assign]
            StarletteRequest.json = recorded_json  # type: ignore[method-assign]
    except ImportError:
        pass


DJANGO_RECORDER = "aiwaf.core.input_recorder.DjangoInputRecorder"


def django_setting() -> Optional[str]:
    """Django's AIWAF_RECORD_INPUTS setting, as text, or None to use the environment."""
    from django.conf import settings

    value = getattr(settings, ENVIRONMENT, None)
    return None if value is None else str(value)


def add_django_recorder() -> bool:
    """Adds the recorder last to MIDDLEWARE when recording is on; whether it's there now.

    Django builds its middleware chain after the apps are ready, so aiwaf's AppConfig.ready
    calls this, and recording needs no MIDDLEWARE line.
    """
    from django.conf import settings

    if destination(django_setting()) is None:
        return False
    middleware = list(getattr(settings, "MIDDLEWARE", None) or [])
    if DJANGO_RECORDER not in middleware:
        settings.MIDDLEWARE = [*middleware, DJANGO_RECORDER]
    return True


class DjangoInputRecorder:
    """Django middleware: records what each view reads. With AIWAF_RECORD_INPUTS (a setting or the
    environment) and aiwaf in INSTALLED_APPS, it's added last to MIDDLEWARE on its own."""

    def __init__(self, get_response):
        self.get_response = get_response
        self.path = destination(django_setting())
        if self.path is not None:
            install()

    def __call__(self, request):
        if self.path is None:
            return self.get_response(request)
        try:
            return self.get_response(request)
        finally:
            recording = getattr(request, "_aiwaf_recording", None)
            _current.set(None)
            if recording is not None:
                match = getattr(request, "resolver_match", None)
                route = getattr(match, "route", None)
                recording.save("/" + route if route is not None else None, request.method, self.path)

    def process_view(self, request, view_func, view_args, view_kwargs):
        # Recording starts here, after other middleware has run, so their reads aren't the view's.
        if self.path is not None:
            recording = Recording(lambda container: container is request.__dict__.get("GET")
                                  or container is getattr(request, "_post", None))
            request._aiwaf_recording = recording
            _current.set(recording)
        return None


def record_flask_inputs(app, path: Optional[str] = None):
    """Flask: records what each view reads, when AIWAF_RECORD_INPUTS (or path) is set."""
    target = destination(path) if path is not None else destination()
    if target is None:
        return app
    install()
    from flask import g, request

    @app.before_request
    def _aiwaf_start_recording():
        actual = request._get_current_object()
        recording = Recording(lambda container: container is actual or container is actual.__dict__.get("args")
                              or container is actual.__dict__.get("form"))
        g._aiwaf_recording = recording
        g._aiwaf_recording_token = _current.set(recording)

    @app.teardown_request
    def _aiwaf_save_recording(_error=None):
        recording = g.pop("_aiwaf_recording", None)
        token = g.pop("_aiwaf_recording_token", None)
        if token is not None:
            try:
                _current.reset(token)
            except ValueError:
                _current.set(None)
        if recording is not None:
            rule = getattr(request, "url_rule", None)
            recording.save(getattr(rule, "rule", None), request.method, target)

    return app


class AsgiInputRecorder:
    """ASGI middleware for FastAPI and Starlette: records what each endpoint reads."""

    def __init__(self, app, path: Optional[str] = None):
        self.app = app
        self.path = destination(path) if path is not None else destination()
        if self.path is not None:
            install()

    async def __call__(self, scope: Dict[str, Any], receive, send):
        if self.path is None or scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        recording = Recording(scope=scope)
        token = _current.set(recording)
        try:
            await self.app(scope, receive, send)
        finally:
            _current.reset(token)
            route = scope.get("route")
            recording.save(getattr(route, "path", None), str(scope.get("method", "GET")), self.path)
