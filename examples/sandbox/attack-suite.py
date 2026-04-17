#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import requests


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ms() -> int:
    return int(time.perf_counter() * 1000)


def _safe_run_id() -> str:
    return _utc_iso().replace(":", "-").replace(".", "-")


def _build_output_path(target_name: str, run_id: str) -> Path:
    return Path(__file__).resolve().parent / f"results_{target_name}_{run_id}.json"


DefaultHeaders = Union[Dict[str, str], Callable[[str, str], Dict[str, str]]]
_default_headers: DefaultHeaders = {}


def set_default_headers(headers: DefaultHeaders) -> None:
    global _default_headers
    _default_headers = headers or {}


def _get_default_headers(method: str, url: str) -> Dict[str, str]:
    if callable(_default_headers):
        return dict(_default_headers(method, url) or {})
    return dict(_default_headers or {})


def _hash_to_octet(seed: str) -> int:
    s = str(seed or "")
    h = 0
    for ch in s:
        h = ((h << 5) - h) + ord(ch)
        h &= 0xFFFFFFFF
    positive = abs(int(h))
    return (positive % 200) + 10


def build_ip(base: str, seed: str, offset: int) -> str:
    octet = (_hash_to_octet(seed) + int(offset)) % 245 + 10
    return f"{base}.{octet}"


def make_ip_generator(base: str, seed: str, start_offset: int) -> Callable[[], str]:
    counter = {"value": int(start_offset)}

    def _next() -> str:
        ip = build_ip(base, seed, counter["value"])
        counter["value"] += 1
        return ip

    return _next


def make_header_generator(static_headers: Dict[str, str], ip_generator: Optional[Callable[[], str]]) -> DefaultHeaders:
    static_headers = dict(static_headers or {})
    if ip_generator is None:
        return static_headers

    def _gen(_method: str, _url: str) -> Dict[str, str]:
        headers = dict(static_headers)
        headers["x-forwarded-for"] = ip_generator()
        return headers

    return _gen


@dataclass(frozen=True)
class RequestResult:
    status: int
    duration_ms: int
    error: Optional[str] = None


def request_once(method: str, url: str, *, headers: Optional[Dict[str, str]] = None, body: Any = None) -> RequestResult:
    start = _now_ms()
    status = 0
    err: Optional[str] = None
    try:
        base_headers = _get_default_headers(method, url)
        merged_headers = {**base_headers, **(headers or {})}

        kwargs: Dict[str, Any] = {
            "method": method,
            "url": url,
            "headers": merged_headers,
            "allow_redirects": False,
            "timeout": 30,
        }
        if body is not None:
            if isinstance(body, (dict, list)):
                kwargs["json"] = body
            else:
                kwargs["data"] = body

        res = requests.request(**kwargs)
        status = int(res.status_code)
        _ = res.text
    except Exception as e:
        status = 0
        err = getattr(e, "code", None) or str(e) or "request_failed"
    end = _now_ms()
    return RequestResult(status=status, duration_ms=end - start, error=err)


def summarize(results: Iterable[RequestResult]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    total_duration = 0
    blocked = 0
    errors = 0
    count = 0

    for r in results:
        count += 1
        status_counts[str(r.status)] = status_counts.get(str(r.status), 0) + 1
        total_duration += int(r.duration_ms)
        if r.status in (403, 405, 409, 429):
            blocked += 1
        if r.status == 0:
            errors += 1

    avg_response_time = (total_duration / count) if count else 0
    return {
        "statusCounts": status_counts,
        "blocked": blocked,
        "avgResponseTime": avg_response_time,
        "errors": errors,
    }


def _delay(ms: int) -> None:
    time.sleep(ms / 1000.0)


def attack_brute_force(base_url: str) -> List[RequestResult]:
    results: List[RequestResult] = []
    url = f"{base_url}/rest/user/login"
    for i in range(50):
        results.append(
            request_once(
                "POST",
                url,
                headers={"content-type": "application/json"},
                body={"email": f"admin{i}@example.com", "password": "password"},
            )
        )
    return results


def attack_credential_stuffing(base_url: str) -> List[RequestResult]:
    results: List[RequestResult] = []
    url = f"{base_url}/rest/user/login"
    candidates = [
        {"email": "admin@juice-sh.op", "password": "admin123"},
        {"email": "admin@juice-sh.op", "password": "password"},
        {"email": "test@juice-sh.op", "password": "test"},
        {"email": "demo@juice-sh.op", "password": "demo"},
    ]
    for cred in candidates:
        for _ in range(10):
            results.append(
                request_once(
                    "POST",
                    url,
                    headers={"content-type": "application/json"},
                    body=cred,
                )
            )
    return results


def attack_path_probe(base_url: str) -> List[RequestResult]:
    paths = [
        "/admin.php",
        "/.env",
        "/.git/config",
        "/../etc/passwd",
        "/wp-login.php",
        "/phpmyadmin",
        "/config.php",
        "/server-status",
        "/actuator/env",
        "/api/internal",
        "/backup.zip",
        "/.well-known/security.txt",
    ]
    return [request_once("GET", f"{base_url}{p}") for p in paths]


def attack_header_probe(base_url: str) -> List[RequestResult]:
    headers = {
        "user-agent": "sqlmap/1.0",
        "x-evil-header": "1",
        "x-forwarded-for": "127.0.0.1",
    }
    return [request_once("GET", f"{base_url}/", headers=headers)]


def attack_header_variations(base_url: str) -> List[RequestResult]:
    uas = [
        "sqlmap/1.8",
        "nikto/2.5.0",
        "masscan/1.3",
        "curl/7.88.1",
        "python-requests/2.31.0",
    ]
    results: List[RequestResult] = []
    for ua in uas:
        results.append(
            request_once(
                "GET",
                f"{base_url}/",
                headers={"user-agent": ua, "x-evil-header": "1"},
            )
        )
    return results


def _run_concurrent(reqs: List[Tuple[str, str, Dict[str, str], Any]]) -> List[RequestResult]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[RequestResult] = []
    with ThreadPoolExecutor(max_workers=min(20, max(1, len(reqs)))) as ex:
        futures = [ex.submit(request_once, m, u, headers=h, body=b) for (m, u, h, b) in reqs]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def attack_burst(base_url: str) -> List[RequestResult]:
    url = f"{base_url}/"
    return _run_concurrent([("GET", url, {}, None) for _ in range(30)])


def attack_burst_mixed(base_url: str) -> List[RequestResult]:
    urls = [f"{base_url}/", f"{base_url}/rest/products", f"{base_url}/rest/user/login"]
    reqs: List[Tuple[str, str, Dict[str, str], Any]] = []
    for i in range(40):
        url = urls[i % len(urls)]
        if url.endswith("/login"):
            reqs.append(
                (
                    "POST",
                    url,
                    {"content-type": "application/json"},
                    {"email": f"burst{i}@example.com", "password": "x"},
                )
            )
        else:
            reqs.append(("GET", url, {}, None))
    return _run_concurrent(reqs)


def attack_method_probe(base_url: str) -> List[RequestResult]:
    return [
        request_once("PUT", f"{base_url}/api/"),
        request_once("DELETE", f"{base_url}/api/"),
        request_once("PATCH", f"{base_url}/api/"),
    ]


def attack_query_injection(base_url: str) -> List[RequestResult]:
    payloads = [
        "/rest/products/search?q=' OR 1=1--",
        "/rest/products/search?q=%3Cscript%3Ealert(1)%3C%2Fscript%3E",
        "/rest/products/search?q=%27%3BWAITFOR%20DELAY%20%270:0:3%27--",
    ]
    return [request_once("GET", f"{base_url}{p}") for p in payloads]


def attack_owasp_top10(base_url: str) -> List[RequestResult]:
    reqs = [
        ("GET", "/rest/products/search?q=' OR 1=1--", {}, None),
        ("GET", "/rest/products/search?q=%3Cscript%3Ealert(1)%3C%2Fscript%3E", {}, None),
        ("GET", "/rest/products/search?q=%7B%22$ne%22:%20null%7D", {}, None),
        ("GET", "/api/Users?filter=__proto__", {}, None),
        ("GET", "/api/Users?filter=%7B%22where%22:%7B%22id%22:1%7D%7D", {}, None),
        ("GET", "/rest/user/whoami", {}, None),
        ("GET", "/admin", {}, None),
        ("GET", "/.env", {}, None),
        ("GET", "/.git/config", {}, None),
        ("GET", "/swagger.json", {}, None),
        ("GET", "/api-docs", {}, None),
        ("GET", "/rest/products/1/reviews", {}, None),
        ("GET", "/rest/products/search?q=%2e%2e%2f%2e%2e%2fetc%2fpasswd", {}, None),
        ("GET", "/rest/user/login?email=admin@juice-sh.op&password=admin123", {}, None),
        ("POST", "/rest/user/login", {"content-type": "application/json"}, {"email": "admin@juice-sh.op", "password": "admin123"}),
        ("POST", "/rest/user/login", {"content-type": "application/json"}, {"email": "admin@juice-sh.op' OR 1=1--", "password": "x"}),
    ]
    return [request_once(m, f"{base_url}{p}", headers=h, body=b) for (m, p, h, b) in reqs]


def attack_long_path(base_url: str) -> List[RequestResult]:
    long_path = "/" + ("a" * 2047)
    return [request_once("GET", f"{base_url}{long_path}")]


def attack_normal_traffic(base_url: str) -> List[RequestResult]:
    results: List[RequestResult] = []
    normal_headers = {
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "accept-language": "en-US,en;q=0.9",
        "accept-encoding": "gzip, deflate, br",
        "connection": "keep-alive",
    }

    results.append(request_once("GET", f"{base_url}/", headers=normal_headers))
    _delay(50)
    results.append(request_once("GET", f"{base_url}/rest/products", headers=normal_headers))
    _delay(50)
    results.append(request_once("GET", f"{base_url}/rest/products/search?q=apple", headers=normal_headers))
    _delay(50)
    results.append(request_once("GET", f"{base_url}/api/Products/1", headers=normal_headers))
    _delay(50)
    results.append(request_once("GET", f"{base_url}/rest/user/whoami", headers=normal_headers))
    _delay(50)
    results.append(request_once("GET", f"{base_url}/api/BasketItems", headers=normal_headers))
    _delay(50)

    user_agents = [
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1",
    ]
    for ua in user_agents:
        headers = dict(normal_headers)
        headers["user-agent"] = ua
        results.append(request_once("GET", f"{base_url}/", headers=headers))
        _delay(50)
        results.append(request_once("GET", f"{base_url}/rest/products", headers=headers))
        _delay(50)

    return results


def run_test_suite(
    base_url: str,
    target_name: str,
    output_file: Optional[Path],
    test_list: List[Tuple[str, Callable[[str], List[RequestResult]]]],
    headers: DefaultHeaders,
) -> Dict[str, Any]:
    set_default_headers(headers)
    health = request_once("GET", f"{base_url}/")
    if health.status == 0:
        reason = f" ({health.error})" if health.error else ""
        raise RuntimeError(f"Unable to reach {base_url}{reason}. Is it running and reachable?")

    run_id = output_file.name.replace("results_", "").replace(".json", "") if output_file else _utc_iso()
    report: Dict[str, Any] = {
        "target": target_name,
        "baseUrl": base_url,
        "runId": run_id,
        "startedAt": _utc_iso(),
        "attacks": [],
    }

    for attack_name, attack_fn in test_list:
        results = attack_fn(base_url)
        summary = summarize(results)
        report["attacks"].append(
            {
                "attack_type": attack_name,
                "requests_sent": len(results),
                "status_counts": summary["statusCounts"],
                "blocked": summary["blocked"],
                "errors": summary["errors"],
                "avg_response_time_ms": summary["avgResponseTime"],
            }
        )

    report["finishedAt"] = _utc_iso()
    if output_file:
        output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved {output_file}")
    return report


def run_normal_traffic_only(base_url: str, target_name: str, output_file: Path, headers: DefaultHeaders) -> Dict[str, Any]:
    return run_test_suite(base_url, target_name, output_file, [("normal_traffic", attack_normal_traffic)], headers)


def run_attacks_suite(base_url: str, target_name: str, output_file: Path, headers: DefaultHeaders) -> Dict[str, Any]:
    tests: List[Tuple[str, Callable[[str], List[RequestResult]]]] = [
        ("brute_force", attack_brute_force),
        ("credential_stuffing", attack_credential_stuffing),
        ("path_probe", attack_path_probe),
        ("header_probe", attack_header_probe),
        ("header_variations", attack_header_variations),
        ("burst", attack_burst),
        ("burst_mixed", attack_burst_mixed),
        ("query_injection", attack_query_injection),
        ("owasp_top10", attack_owasp_top10),
        ("long_path", attack_long_path),
        ("method_probe", attack_method_probe),
    ]
    return run_test_suite(base_url, target_name, output_file, tests, headers)


def run_default_comparison() -> None:
    run_id = _safe_run_id()
    run_seed = f"run-{run_id}"
    targets = [
        {"name": "direct", "url": "http://localhost:3001"},
        {"name": "protected_django", "url": "http://localhost:3009"},
        {"name": "protected_flask", "url": "http://localhost:3010"},
        {"name": "protected_fastapi", "url": "http://localhost:3011"},
    ]

    all_reports: Dict[str, List[Dict[str, Any]]] = {"normal": [], "attacks": []}

    for idx, target in enumerate(targets):
        target["normalIp"] = build_ip("198.51.100", run_seed, idx)
        target["attackIp"] = build_ip("203.0.113", run_seed, idx + 50)
        target["normalIpGenerator"] = make_ip_generator("198.51.100", run_seed, idx + 100)
        target["attackIpGenerator"] = make_ip_generator("203.0.113", run_seed, idx + 200)

    for target in targets:
        normal_output = _build_output_path(f"{target['name']}_normal", run_id)
        print(f"\nRunning normal traffic tests for {target['name']}...")
        normal_report = run_normal_traffic_only(
            target["url"],
            target["name"],
            normal_output,
            make_header_generator({"x-forwarded-for": target["normalIp"]}, target["normalIpGenerator"]),
        )
        all_reports["normal"].append(normal_report)

        attacks_output = _build_output_path(f"{target['name']}_attacks", run_id)
        print(f"\nRunning attack tests for {target['name']}...")
        attacks_report = run_attacks_suite(
            target["url"],
            target["name"],
            attacks_output,
            make_header_generator({"x-forwarded-for": target["attackIp"]}, target["attackIpGenerator"]),
        )
        all_reports["attacks"].append(attacks_report)

    comparison_file = Path(__file__).resolve().parent / f"comparison_modes_{run_id}.json"
    comparison_file.write_text(
        json.dumps({"runId": run_id, "generatedAt": _utc_iso(), "normal": all_reports["normal"], "attacks": all_reports["attacks"]}, indent=2),
        encoding="utf-8",
    )
    print(f"\n Saved comprehensive comparison: {comparison_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AIWAF sandbox attack suite (Python)")
    parser.add_argument("base_url", nargs="?", help="Target base URL (e.g. http://localhost:3009)")
    parser.add_argument("target_name", nargs="?", default="target", help="Label for the target")
    parser.add_argument("--mode", default="all", choices=["normal", "attacks", "all"])
    args = parser.parse_args()

    if not args.base_url:
        run_default_comparison()
        return

    run_id = _safe_run_id()
    output_file = _build_output_path(f"{args.target_name}_{args.mode}", run_id)
    run_seed = f"{args.target_name}-{run_id}"
    normal_ip = build_ip("198.51.100", run_seed, 1)
    attack_ip = build_ip("203.0.113", run_seed, 2)
    normal_ip_gen = make_ip_generator("198.51.100", run_seed, 100)
    attack_ip_gen = make_ip_generator("203.0.113", run_seed, 200)

    if args.mode == "normal":
        run_normal_traffic_only(args.base_url, args.target_name, output_file, make_header_generator({"x-forwarded-for": normal_ip}, normal_ip_gen))
        return
    if args.mode == "attacks":
        run_attacks_suite(args.base_url, args.target_name, output_file, make_header_generator({"x-forwarded-for": attack_ip}, attack_ip_gen))
        return

    run_test_suite(
        args.base_url,
        args.target_name,
        output_file,
        [
            ("normal_traffic", attack_normal_traffic),
            ("brute_force", attack_brute_force),
            ("credential_stuffing", attack_credential_stuffing),
            ("path_probe", attack_path_probe),
            ("header_probe", attack_header_probe),
            ("header_variations", attack_header_variations),
            ("burst", attack_burst),
            ("burst_mixed", attack_burst_mixed),
            ("query_injection", attack_query_injection),
            ("owasp_top10", attack_owasp_top10),
            ("long_path", attack_long_path),
            ("method_probe", attack_method_probe),
        ],
        make_header_generator({"x-forwarded-for": normal_ip}, normal_ip_gen),
    )


if __name__ == "__main__":
    main()
