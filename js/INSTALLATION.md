# AIWAF-JS Installation Guide

This guide covers local setup, Redis setup, training setup, and common failure modes.

## Prerequisites

- Node.js 18+ recommended
- npm 9+ recommended
- Build tooling required by `sqlite3` (platform dependent)

## 1. Install Package

For application use:

```bash
npm install aiwaf-js
```

For local development in this repository:

```bash
npm install
```

## 2. Basic Integration (Express)

```js
const express = require('express');
const aiwaf = require('aiwaf-js');

const app = express();
app.use(express.json());

app.use(aiwaf({
  staticKeywords: ['.php', '.env', '.git'],
  dynamicTopN: 10,
  WINDOW_SEC: 10,
  MAX_REQ: 20,
  FLOOD_REQ: 40,
  HONEYPOT_FIELD: 'hp_field',
  uuidRoutePrefix: '/user'
}));
```

## 3. Optional Redis Setup

Set `REDIS_URL` (or `AIWAF_REDIS_URL`) before app startup:

```bash
export REDIS_URL=redis://localhost:6379
```

PowerShell:

```powershell
$env:REDIS_URL = 'redis://localhost:6379'
```

If Redis is not configured or not reachable, AIWAF-JS falls back to in-memory behavior.

## 3.1 Optional GeoIP MMDB Setup

Install MMDB reader:

```bash
npm install maxmind
```

Place your database at `geolock/ipinfo_lite.mmdb` or set:

```bash
export AIWAF_GEO_MMDB_PATH=/absolute/path/to/ipinfo_lite.mmdb
```

## 4. Train a Model from Logs

By default, trainer reads `/var/log/nginx/access.log`. Override as needed:

```bash
NODE_LOG_PATH=/path/to/access.log npm run train
```

Include rotated logs:

```bash
NODE_LOG_GLOB='/path/to/access.log.*' npm run train
```

Output model artifact:

- `resources/model.json`

## 5. Verify Installation

Run tests:

```bash
npm test
```

Check CLI wiring:

```bash
npm run aiwaf -- help
```

Run a minimal app and hit a known benign route (`/`) and a suspicious route (for example path with `.php`) to validate block behavior.

## 6. Troubleshooting

### `Failed to load pretrained model`

- Run `npm run train` to generate `resources/model.json`.
- Ensure process has read access to the `resources/` directory.

### Redis warnings or connection failures

- Verify `REDIS_URL` value and Redis server health.
- Runtime is designed to continue with fallback behavior.

### SQLite errors (`blocked_ips` table not found)

- Ensure process can create/write `./aiwaf.sqlite`.
- `blacklistManager` auto-initializes the table, but write permissions are required.
- If DB logging is unavailable, enable CSV middleware logs:
  - `AIWAF_MIDDLEWARE_LOG_CSV=true`
  - `AIWAF_MIDDLEWARE_LOG_CSV_PATH=logs/aiwaf-requests.csv`
- Core tables also fall back automatically to CSV files in `logs/storage/` when DB operations fail.

### `sqlite3` install/build issues

- Install platform-native build dependencies and reinstall packages.
- On CI/container images, ensure compiler toolchain is present.

### Training finds no logs

- Confirm `NODE_LOG_PATH` exists and is readable.
- If using rotation, set `NODE_LOG_GLOB` to a valid glob.

## 7. Production Notes

- Prefer Redis or a custom shared cache backend for multi-instance deployments.
- Place middleware after body parsing middleware if honeypot detection is required.
- Review rate limits and thresholds against real traffic profiles before broad rollout.
