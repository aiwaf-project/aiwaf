import os
import subprocess
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class DockerRedis:
    container_id: str
    host: str
    port: int

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/0"


def _run(args, *, timeout=60):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout)


def start_docker_redis(*, image: str = "redis:7-alpine") -> DockerRedis:
    """
    Start a temporary Redis container and return a reachable URL.
    Uses `-P` to map container port 6379 to a random host port.
    """
    cid = _run(["docker", "run", "-d", "-P", image]).stdout.strip()
    try:
        # Discover mapped port, output example: "0.0.0.0:49153"
        port_out = _run(["docker", "port", cid, "6379/tcp"]).stdout.strip().splitlines()[0].strip()
        hostport = port_out.rsplit(":", 1)[-1]
        port = int(hostport)

        info = DockerRedis(container_id=cid, host="127.0.0.1", port=port)
        _wait_for_redis(info.url)
        return info
    except Exception:
        stop_docker_container(cid)
        raise


def stop_docker_container(container_id: str) -> None:
    try:
        subprocess.run(["docker", "rm", "-f", container_id], check=False, capture_output=True, text=True, timeout=30)
    except Exception:
        return


def _wait_for_redis(redis_url: str, *, timeout_seconds: float = 15.0) -> None:
    deadline = time.time() + timeout_seconds
    last_exc = None
    while time.time() < deadline:
        try:
            # Use AIWAF's own redis client fallback (no redis-py required).
            from aiwaf.core.cache_backend import _redis_from_url

            client = _redis_from_url(redis_url)
            if client.ping():
                return
        except Exception as exc:
            last_exc = exc
            time.sleep(0.2)
    raise RuntimeError(f"Redis did not become ready: {last_exc}")
