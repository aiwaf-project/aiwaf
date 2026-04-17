# AIWAF (Python) + OWASP Juice Shop Sandbox

This sandbox runs AIWAF in front of OWASP Juice Shop using Django and Flask proxy apps.

## Run

From `examples/sandbox/`:

```bash
docker compose up --build
```

Then open:

- AIWAF-protected (Django): `http://localhost:3009`
- AIWAF-protected (Flask): `http://localhost:3010`
- Direct Juice Shop: `http://localhost:3001`

## Test

```bash
curl http://localhost:3009
curl http://localhost:3009/admin.php
curl http://localhost:3009/../../etc/passwd
curl -A "sqlmap/1.0" http://localhost:3009
```

Check logs in the `aiwaf_logs` volume.

## Attack Suite

Run against direct Juice Shop:

```bash
python attack-suite.py http://localhost:3001 direct --mode=attacks
```

Run against AIWAF-protected Juice Shop (Django):

```bash
python attack-suite.py http://localhost:3009 protected_django --mode=attacks
```

Run against AIWAF-protected Juice Shop (Flask):

```bash
python attack-suite.py http://localhost:3010 protected_flask --mode=attacks
```

Compare results:

```bash
python compare-results.py results_direct_*.json results_protected_django_*.json results_protected_flask_*.json
```

Or run the full suite + comparison in one command:

```bash
python run-and-compare.py
```
