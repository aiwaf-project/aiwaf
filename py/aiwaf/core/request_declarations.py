"""Read request declarations only; never inspect request data or run views."""
import inspect
import re

BODY_METHODS = {'POST', 'PUT', 'PATCH', 'DELETE'}
MEDIA = re.compile(r'[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+')


def _media(value):
    return value if isinstance(value, str) and len(value) <= 127 and MEDIA.fullmatch(value) else None


def _rows(methods, media=None, media_source=None, limit=None, limit_source=None):
    return [{'method': m, 'media_types': media, 'media_types_source': media_source,
             'max_body_bytes': limit, 'max_body_bytes_source': limit_source}
            for m in sorted(set(methods) & BODY_METHODS)]


def django_declarations(callback, methods):
    media = None
    try:
        from rest_framework.views import APIView
        from rest_framework.viewsets import ViewSetMixin
        from rest_framework.negotiation import DefaultContentNegotiation
    except ImportError:
        return _rows(methods)
    cls = getattr(callback, 'cls', None) or getattr(callback, 'view_class', None)
    if not isinstance(cls, type) or not issubclass(cls, APIView):
        return _rows(methods)
    # Dynamic parser selection and custom request initialization are unknown.
    if (cls.get_parsers is not APIView.get_parsers or
            cls.get_content_negotiator is not APIView.get_content_negotiator or
            cls.initialize_request not in {APIView.initialize_request, ViewSetMixin.initialize_request}):
        return _rows(methods)
    overrides = getattr(callback, 'initkwargs', {}) or getattr(callback, 'view_initkwargs', {}) or {}
    negotiation = overrides.get('content_negotiation_class', cls.content_negotiation_class)
    parsers = overrides.get('parser_classes', cls.parser_classes)
    if negotiation is DefaultContentNegotiation and isinstance(parsers, (list, tuple)) and 1 <= len(parsers) <= 32:
        values = [inspect.getattr_static(parser, 'media_type', None) if isinstance(parser, type) else None for parser in parsers]
        # Wildcard/custom-dynamic parsers make the complete declaration unknown.
        if all(_media(v) for v in values):
            media = sorted(set(values))
    return _rows(methods, media, 'drf_parsers' if media else None)


def flask_declarations(app, methods):
    # Per-field memory limits and runtime request overrides are not body caps.
    limit = app.config.get('MAX_CONTENT_LENGTH')
    if type(limit) is not int or not 1 <= limit <= 1073741824:
        limit = None
    return _rows(methods, limit=limit, limit_source='flask_max_content_length' if limit else None)


def fastapi_declarations(route, methods):
    from fastapi.routing import APIRoute
    from fastapi.params import Body
    media = None
    if isinstance(route, APIRoute) and type(route).get_route_handler is APIRoute.get_route_handler:
        info = getattr(getattr(route, 'body_field', None), 'field_info', None)
        if isinstance(info, Body):
            value = _media(info.media_type)
            if value:
                media = [value]
    # Body(max_length=...) and form part limits are not whole-request byte caps.
    return _rows(methods, media, 'fastapi_body' if media else None)


def merge_entries(previous, current):
    """Keep per-method declarations when multiple routes share a path."""
    if previous is None:
        return current
    result = {**current, 'methods': sorted(set(previous['methods']) | set(current['methods']))}
    # Every view serving the path, and the methods each serves, so each one's inputs are linked
    # to it and to its methods.
    view_methods = {}
    declared_methods = {}
    declared_types = {}
    declared_locations = {}
    for entry in (previous, current):
        for view, methods in (entry.get('view_methods') or {entry.get('view'): entry.get('methods') or []}).items():
            if view:
                view_methods[str(view)] = sorted(set(view_methods.get(str(view), [])) | set(methods))
        known = entry.get('declared_methods')
        if known is None and isinstance(entry.get('declared_inputs'), dict):
            names = sorted((entry['declared_inputs'].get('inputs') or {}).keys())
            known = {method: names for method in entry.get('methods') or []}
        for method, names in (known or {}).items():
            declared_methods[method] = sorted(set(declared_methods.get(method, [])) | set(names))
        # Each method's own types, kept apart from another method's on the same path; two
        # handlers for one method that type a name differently leave it untyped for it.
        types = entry.get('declared_types')
        if types is None and isinstance(entry.get('declared_inputs'), dict):
            typed = {name: kind for name, kind in (entry['declared_inputs'].get('inputs') or {}).items() if kind and kind != 'text'}
            types = {str(method).upper(): dict(typed) for method in entry.get('methods') or []} if typed else {}
        for method, own in (types or {}).items():
            merged = declared_types.setdefault(method, {})
            for name, kind in own.items():
                merged[name] = kind if merged.get(name, kind) == kind else 'text'
        # And each method's types per location, from each handler's framework list.
        locations = entry.get('declared_locations')
        if locations is None and isinstance(entry.get('declared_inputs'), dict) and entry['declared_inputs'].get('locations'):
            locations = {str(method).upper(): entry['declared_inputs']['locations'] for method in entry.get('methods') or []}
        for method, by_where in (locations or {}).items():
            into = declared_locations.setdefault(method, {})
            for where, types in by_where.items():
                merged = into.setdefault(where, {})
                for name, kind in types.items():
                    merged[name] = kind if merged.get(name, kind) == kind else 'text'
    if len(view_methods) > 1:
        result['views'] = sorted(view_methods)
        result['view_methods'] = view_methods
    if declared_methods:
        result['declared_methods'] = declared_methods
    if declared_types:
        result['declared_types'] = declared_types
    if declared_locations:
        result['declared_locations'] = declared_locations
    if previous.get('declared_inputs') or current.get('declared_inputs'):
        from aiwaf.core.input_sources import combine

        result['declared_inputs'] = combine(previous.get('declared_inputs'), current.get('declared_inputs'))
    grouped = {}
    for entry in (previous, current):
        known = {row['method']: row for row in entry.get('request_constraints', [])}
        for fallback in _rows(entry['methods']):
            row = known.get(fallback['method'], fallback)
            grouped.setdefault(row['method'], []).append(row)
    rows = []
    for method, group in sorted(grouped.items()):
        merged = _rows([method])[0]
        # Disagreement/unknown is kept unknown, never resolved by choosing a
        # narrower media type or a smaller byte ceiling.
        for field in ('media_types', 'max_body_bytes'):
            value = group[0][field]
            if value is not None and all(row[field] == value for row in group):
                merged[field] = value
                merged[field + '_source'] = group[0][field + '_source']
        rows.append(merged)
    result['request_constraints'] = rows
    return result


def mask_shadowed_declarations(routes, extra_prefixes=()):
    """An overlapping dynamic route/mount makes a literal contract uncertain."""
    patterns = []
    markers = r'<>{}*^$\?#[]()|+'
    for path, entry in routes.items():
        position = next((i for i, char in enumerate(path) if char in markers), None)
        if position is not None:
            patterns.append((path[:position], set(entry['methods']) & BODY_METHODS))
    patterns.extend((prefix, BODY_METHODS) for prefix in extra_prefixes)
    for path, entry in routes.items():
        if any(char in path for char in markers):
            continue
        for row in entry.get('request_constraints', []):
            if any(path.startswith(prefix) and row['method'] in methods for prefix, methods in patterns):
                row.update(media_types=None, media_types_source=None, max_body_bytes=None, max_body_bytes_source=None)
    return routes
