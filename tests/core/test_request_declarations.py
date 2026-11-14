"""Declarations from real framework objects; handlers are never invoked."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_flask_body_ceiling_is_distinct_from_form_field_memory_and_no_media_is_guessed():
    from flask import Flask
    from aiwaf.flask.path_manifest import extract_flask_routes
    app = Flask(__name__)
    @app.post('/submit')
    def submit():
        raise AssertionError('must not execute')
    app.config['MAX_FORM_MEMORY_SIZE'] = 123
    rows = extract_flask_routes(app)['/submit']['request_constraints']
    assert rows[0]['max_body_bytes'] is None
    assert rows[0]['media_types'] is None
    app.config['MAX_CONTENT_LENGTH'] = 2048
    row = extract_flask_routes(app)['/submit']['request_constraints'][0]
    assert row['max_body_bytes'] == 2048
    assert row['max_body_bytes_source'] == 'flask_max_content_length'
    app.config['MAX_CONTENT_LENGTH'] = True
    assert extract_flask_routes(app)['/submit']['request_constraints'][0]['max_body_bytes'] is None


def test_fastapi_json_form_file_and_plain_request_declarations():
    from fastapi import FastAPI, Body, Form, File, Request, UploadFile
    from aiwaf.fast.path_manifest import extract_fastapi_routes
    app = FastAPI()
    @app.post('/json')
    def json_body(value: dict = Body()):
        raise AssertionError('must not execute')
    @app.post('/form')
    def form(value: str = Form()):
        raise AssertionError('must not execute')
    @app.post('/file')
    def upload(value: UploadFile = File()):
        raise AssertionError('must not execute')
    @app.post('/text')
    def text(value: str = Body(media_type='text/plain', max_length=10)):
        return {}
    @app.post('/unknown')
    def dynamic(request: Request):
        return {'response': 'does not declare JSON input'}
    routes = extract_fastapi_routes(app)
    expected = {'/json': 'application/json', '/form': 'application/x-www-form-urlencoded',
                '/file': 'multipart/form-data', '/text': 'text/plain'}
    for path, media in expected.items():
        row = routes[path]['request_constraints'][0]
        assert row['media_types'] == [media]
        assert row['media_types_source'] == 'fastapi_body'
        assert row['max_body_bytes'] is None
    assert routes['/unknown']['request_constraints'][0]['media_types'] is None
    # No body field names, schemas, defaults or values are exported here.
    assert 'value' not in json.dumps([r['request_constraints'] for r in routes.values()])


def test_fastapi_shared_paths_keep_per_method_facts_and_conflicts_unknown():
    from fastapi import FastAPI, Body, Form
    from aiwaf.fast.path_manifest import extract_fastapi_routes
    app = FastAPI()
    app.post('/shared')(lambda value=Body(): {})
    app.put('/shared')(lambda value=Form(): {})
    app.post('/conflict')(lambda value=Body(): {})
    app.post('/conflict')(lambda value=Form(): {})
    routes = extract_fastapi_routes(app)
    assert routes['/shared']['methods'] == ['POST', 'PUT']
    rows = {r['method']: r for r in routes['/shared']['request_constraints']}
    assert rows['POST']['media_types'] == ['application/json']
    assert rows['PUT']['media_types'] == ['application/x-www-form-urlencoded']
    assert routes['/conflict']['request_constraints'][0]['media_types'] is None


def test_fastapi_dynamic_route_or_mount_does_not_prove_a_literal_contract():
    from fastapi import FastAPI, Body
    from aiwaf.fast.path_manifest import extract_fastapi_routes
    app = FastAPI()
    app.post('/items/{item}')(lambda item, value=Body(): {})
    app.post('/items/create')(lambda value=Body(): {})
    app.mount('/mounted', FastAPI())
    app.post('/mounted/create')(lambda value=Body(): {})
    routes = extract_fastapi_routes(app)
    assert routes['/items/create']['request_constraints'][0]['media_types'] is None
    assert routes['/mounted/create']['request_constraints'][0]['media_types'] is None


def test_drf_parsers_actions_dynamic_and_wildcards_in_isolated_process():
    script = r'''
import django
from django.conf import settings
settings.configure(SECRET_KEY='declaration-test', ROOT_URLCONF='__main__', INSTALLED_APPS=['django.contrib.auth', 'django.contrib.contenttypes', 'rest_framework'])
django.setup()
from django.urls import path
from django.test import override_settings
from rest_framework.views import APIView
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser, FileUploadParser
from rest_framework.decorators import api_view, parser_classes, action
from rest_framework.viewsets import ViewSet
from rest_framework.routers import SimpleRouter
from aiwaf.django.path_manifest import extract_django_routes
class Json(APIView):
    parser_classes = [JSONParser]
    def post(self, request): raise AssertionError('must not execute')
class Forms(Json): parser_classes = [FormParser, MultiPartParser]
class Wild(Json): parser_classes = [JSONParser, FileUploadParser]
class Dynamic(Json):
    def get_parsers(self): raise AssertionError('must not execute')
@api_view(['POST'])
@parser_classes([JSONParser])
def decorated(request): raise AssertionError('must not execute')
class Actions(ViewSet):
    @action(detail=False, methods=['post'], parser_classes=[MultiPartParser])
    def upload(self, request): raise AssertionError('must not execute')
router = SimpleRouter()
router.register('actions', Actions, basename='actions')
urlpatterns = [path('json/', Json.as_view()), path('forms/', Forms.as_view()), path('wild/', Wild.as_view()), path('dynamic/', Dynamic.as_view()), path('decorated/', decorated)] + router.urls
with override_settings(DATA_UPLOAD_MAX_MEMORY_SIZE=1234, FILE_UPLOAD_MAX_MEMORY_SIZE=456):
    routes = extract_django_routes()
    assert routes['/json/']['request_constraints'][0]['media_types'] == ['application/json']
    assert routes['/forms/']['request_constraints'][0]['media_types'] == ['application/x-www-form-urlencoded', 'multipart/form-data']
    assert routes['/wild/']['request_constraints'][0]['media_types'] is None
    assert routes['/dynamic/']['request_constraints'][0]['media_types'] is None
    assert routes['/decorated/']['request_constraints'][0]['media_types'] == ['application/json']
    assert routes['/actions/upload/']['request_constraints'][0]['media_types'] == ['multipart/form-data']
    assert all(row['max_body_bytes'] is None for route in routes.values() for row in route['request_constraints'])
'''
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2] / 'py') + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
