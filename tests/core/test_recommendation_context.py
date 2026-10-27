"""Real framework objects; no application names or request values leave adapters."""
import json
import pytest
from bloud.context import application_context


def test_flask_blueprints_methods_and_private_literals():
    from flask import Flask, Blueprint
    app = Flask(__name__)
    blueprint = Blueprint('private_person_alice', __name__, url_prefix='/api')
    blueprint.add_url_rule('/users/<email>/', view_func=lambda email: '', methods=['GET'])
    blueprint.add_url_rule('/alice@example.com/', view_func=lambda: '', endpoint='personal', methods=['POST'])
    app.register_blueprint(blueprint)
    value = application_context('flask', app)
    assert all(r['methods_declared'] for r in value['routes'])
    assert any('HEAD' in r['methods'] and 'OPTIONS' in r['methods'] for r in value['routes'])
    assert 'alice' not in json.dumps(value)
    assert 'email' not in json.dumps(value)
    assert all(r['auth'] == 'unknown' for r in value['routes'])


def test_fastapi_dependency_is_not_authentication_and_security_is_recognized():
    from fastapi import FastAPI, Depends
    from fastapi.security import HTTPBearer
    from pydantic import BaseModel
    app = FastAPI()
    class Body(BaseModel):
        private_email: str
    def ordinary():
        return 1
    @app.post('/api/users/')
    def create(body: Body, dependency=Depends(ordinary)):
        return {}
    @app.get('/profile/')
    def profile(token=Depends(HTTPBearer())):
        return {}
    @app.get('/private/')
    def optional(token=Depends(HTTPBearer(auto_error=False))):
        return {}
    rows = {r['template']: r for r in application_context('fastapi', app)['routes']}
    assert rows['/api/users/']['dependencies'] is True
    assert rows['/api/users/']['auth'] == 'unknown'
    assert rows['/api/users/']['body_validation'] == 'declared'
    assert rows['/profile/']['auth'] == 'declared'
    assert rows['/private/']['auth'] == 'unknown'
    assert 'private_email' not in json.dumps(rows)


def test_fastapi_mounted_app_prevents_unproven_literal_edge_candidates():
    from fastapi import FastAPI
    app, mounted = FastAPI(), FastAPI()
    app.mount('/api', mounted)
    @app.get('/api/users/')
    def users():
        return {}
    rows = {r['template']: r for r in application_context('fastapi', app)['routes']}
    assert rows['/api/users/']['literal'] is False


def test_django_declared_methods_csrf_and_decorator_auth():
    from django.conf import settings
    if not settings.configured:
        settings.configure(SECRET_KEY='context-test', ROOT_URLCONF=__name__, INSTALLED_APPS=[], MIDDLEWARE=[], DEBUG=False, ALLOWED_HOSTS=['testserver'])
    from django.test import override_settings
    from django.urls import path, re_path
    from django.views.decorators.http import require_POST
    from django.views.decorators.csrf import csrf_exempt
    from django.contrib.auth.decorators import login_required
    from django.http import HttpResponse
    @csrf_exempt
    @require_POST
    def post(request):
        return HttpResponse('')
    @login_required
    def secured(request):
        return HttpResponse('')
    global urlpatterns
    urlpatterns = [path('api/users/', post), path('profile/', secured),
        re_path(r'^private/(?P<email>.+)/$', post)]
    with override_settings(ROOT_URLCONF=__name__, MIDDLEWARE=['django.middleware.csrf.CsrfViewMiddleware']):
        rows = {r['template']: r for r in application_context('django')['routes']}
    assert rows['/api/users/']['methods'] == ['POST']
    assert rows['/api/users/']['methods_declared'] is True
    assert rows['/api/users/']['csrf'] == 'exempt'
    assert rows['/profile/']['auth'] == 'declared'
    assert '/{route}' in rows
    assert 'email' not in json.dumps(rows)


def test_real_drf_and_django_class_views_in_isolated_django_process():
    import subprocess
    import sys
    pytest.importorskip('rest_framework')
    script = '''
import django
from django.conf import settings
settings.configure(SECRET_KEY='adapter-test', ROOT_URLCONF='__main__', DEBUG=False,
    INSTALLED_APPS=['django.contrib.auth','django.contrib.contenttypes','rest_framework'], MIDDLEWARE=[])
django.setup()
from django.urls import path
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin
from rest_framework.views import APIView
from rest_framework.viewsets import ViewSet
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.throttling import UserRateThrottle
from bloud.context import django_context
class Secure(APIView):
    permission_classes=[IsAuthenticated]
    throttle_classes=[UserRateThrottle]
    def post(self, request): pass
class Public(APIView):
    permission_classes=[AllowAny]
    def get(self, request): pass
class PrivatePage(LoginRequiredMixin, View):
    def get(self, request): pass
class Users(ViewSet):
    permission_classes=[IsAuthenticated]
    def list(self, request): pass
urlpatterns=[path('api/users/',Secure.as_view()),path('public/',Public.as_view()),
    path('profile/',PrivatePage.as_view()),path('users/',Users.as_view({'get':'list'}))]
rows={r['template']:r for r in django_context()['routes']}
assert rows['/api/users/']['auth']=='declared'
assert rows['/api/users/']['throttling']=='declared'
assert rows['/api/users/']['csrf']=='unknown'
assert rows['/public/']['auth']=='public'
assert rows['/profile/']['auth']=='declared'
assert rows['/users/']['methods_declared'] is True
assert 'HEAD' in rows['/users/']['methods']
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
