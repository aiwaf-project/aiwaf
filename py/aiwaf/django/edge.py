from pathlib import Path
from urllib.parse import urlsplit
from django.conf import settings
from django.http import JsonResponse
from aiwaf.core.edge_bridge import create_agent


class EdgeAssistedMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        path = getattr(settings, 'AIWAF_BOUVANE_CONFIG', Path(getattr(settings, 'BASE_DIR', '.')) / '.bouvane.json')
        self.agent = create_agent(path, framework='django')
        if self.agent:
            self.agent.start()

    def __call__(self, request):
        if self.agent:
            self.agent.start()
            token = request.META.pop('HTTP_X_BOUVANE_ORIGIN_TOKEN', '')
            ip = request.META.get('HTTP_CF_CONNECTING_IP', '') if self.agent.trusted_origin(token) else request.META.get('REMOTE_ADDR', '')
            if self.agent.blocked(hostname=urlsplit('//' + request.get_host()).hostname or '', path=request.path_info,
                                  method=request.method, ip=ip, origin_token=token):
                return JsonResponse({'error': 'Forbidden'}, status=403)
        return self.get_response(request)
