# AIWAF-JS + OWASP Juice Shop Sandbox

This sandbox runs AIWAF-JS as a proxy in front of OWASP Juice Shop.

## Run

From `examples/sandbox/`:

```bash
docker compose up --build
```

Then open:

- AIWAF-protected: `http://localhost:3000`
- AIWAF-protected (Fastify): `http://localhost:3002`
- AIWAF-protected (Hapi): `http://localhost:3003`
- AIWAF-protected (Koa): `http://localhost:3004`
- AIWAF-protected (NestJS): `http://localhost:3005`
- AIWAF-protected (Next.js): `http://localhost:3006`
- AIWAF-protected (AdonisJS): `http://localhost:3007`
- AIWAF-protected (Sails.js): `http://localhost:3008`
- Direct Juice Shop: `http://localhost:3001`

## Test

```bash
curl http://localhost:3000
curl http://localhost:3000/admin.php
curl http://localhost:3000/../../etc/passwd
curl -A "sqlmap/1.0" http://localhost:3000
```

Check logs in the `aiwaf_logs` volume (JSONL).
