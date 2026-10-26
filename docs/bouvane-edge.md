# Optional Bouvane edge-assisted mode

AIWAF can discover the separate Bouvane Python package and synchronize application
context after process startup/reload. Installation alone changes no enforcement.
All existing anomaly, UUID, rate-limit, keyword, header, and other runtime
middleware remains enabled. This release delegates only explicit literal path/IP
**block** policies for which Bouvane can verify equivalent edge protection.

## Configure an application

Install the updated AIWAF and Bouvane checkouts in the application's environment.
The Bouvane website must also run the updated backend and migrations.

```powershell
python -m pip install -e ../aiwaf
python -m pip install -e '../bcloud[keyring]'
bouvane login
```

Run login from the application directory. Choose browser sign-in or access-key
login. Browser authorization lets
you create or select an application and enter its name, hostname, framework, and
environment. Connect Cloudflare on the website, then approve the terminal code
for that application. The CLI stores an application-scoped token in the OS
keyring and automatically writes the local project link; `bouvane init` is not
required. Use `bouvane login --no-browser` on a headless system.

Alternatively, create an application access key under **Access keys** on the
website, then run `bouvane login --method key` and paste it into the hidden prompt.
The key is validated against its application before the CLI stores it. Both
methods produce the same local link and background-sync configuration.

Reload your application after linking. AIWAF discovers the registered
project and automatically starts background context synchronization; no separate
enable command or policy JSON file is required. Older registered configurations
without an `edge` section receive the same synchronization-only default when
Bouvane is installed. Explicitly disabled/local configurations are respected.
Use `bouvane edge --local` after login to retain local enforcement without sync.

Choose Django, Flask, or FastAPI in the web setup form. Optionally
attach additional explicit edge blocks with
`bouvane edge --enable --explicit edge-policies.json`. That file uses the package's
existing normalized JSON list:

```json
[
  {
    "id": "retired_endpoint",
    "type": "path_block",
    "match": {"path": "/retired/", "methods": ["GET", "POST"]},
    "configuration": {},
    "action": "block",
    "source": "explicit"
  }
]
```

The edge bridge reads `.bouvane.json`; registration enables context sync without
storing credentials. Additional explicit policies are optional and are not
invented from route names.
Reload the application after changing its configuration. Explicit policy file
changes are picked up by background refresh as well as by process reload.

## Framework startup hooks

* Django: with `aiwaf.django` installed, AppConfig discovers an enabled project
  configuration under `BASE_DIR` and installs the edge middleware first. Its
  construction starts the background agent after Django setup. Management checks
  and the development autoreloader parent do not construct the request handler.
* Flask: the standard `AIWAF(app)` / `init_app(app)` registration installs a
  request gate before other AIWAF hooks and starts a debounced background export.
  The periodic refresh captures routes registered later by an application factory.
* FastAPI: the standard `AIWAF(app)` adds an outer ASGI gate. The agent starts
  when the server constructs the middleware stack, after application registration.

For custom locations set Django/Flask `AIWAF_BOUVANE_CONFIG` to the JSON path or
pass `AIWAF_BOUVANE_CONFIG="/path/.bouvane.json"` to FastAPI's AIWAF configuration.
The native CSV entry points do not install this bridge automatically.

Each process debounces startup, exports through AIWAF's existing route-analysis
interfaces, normalizes the minimal bundle, and compares fingerprints with the
server. Only added/modified policy entries and deleted IDs are pushed after the
first normalized baseline; unchanged reloads push nothing. Source files, handler
source, and the raw manifest are not uploaded. Changed pushes create parented
policy commits that can be reviewed and deployed independently on the website.
Network and keyring operations run on a daemon thread, never on the request
path. Failures clear edge delegation and retry with bounded exponential backoff.
Default refresh is 30 seconds; `edge.refresh_seconds` is bounded to 10–300.
Repeated startup hooks share one agent within a framework application; the
agent resets its thread and lease after a fork. Multiple workers may export the
same context, but server revision fingerprints and deployment idempotency keys
prevent duplicate revisions/deployments.

## Prove edge coverage before delegating

Configure Cloudflare to **overwrite** the origin request header
`X-Bouvane-Origin-Token` with a private, random value, and set the identical value
in the application's `BOUVANE_ORIGIN_SECRET` environment variable. Use at least
32 random characters; generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
Use HTTPS to the API and origin, preserve the original hostname, and restrict
direct origin access using your infrastructure's firewall/tunnel controls.
The origin-header configuration is a separate infrastructure setup step; this
release does not provision Cloudflare Transform Rules automatically.

The bridge strips this secret header before passing requests to the remaining
middleware/application. Forwarded client-IP information is used by the bridge
only when that origin secret matches. Without a matching secret, requests keep
local explicit-policy enforcement, regardless of any forged Cloudflare headers.

Bouvane's `/api/v1/edge-status` endpoint must verify all of the following:

* The caller is authorized for the application.
* The latest desired bundle fingerprint matches the running application.
* A successful deployment covers that exact revision and equivalent rule set.
* No deployment is queued, running, or uncertain; the integration is active.
* The currently observed Cloudflare rules match the verified deployment.

Only then does it issue a 30-second lease with per-policy fingerprints. The
application checks the lease, hostname, exact policy fingerprint, and origin
proof on each request before skipping its corresponding explicit local block.
This does not bypass any existing AIWAF middleware component. If any condition
fails, local fallback applies. Challenge actions and unsupported explicit policy
types are rejected in edge-assisted fallback configuration.

Managed deployments stop new leases and wait for existing leases to expire
before mutating provider rules. Manual/out-of-band Cloudflare changes cannot be
detected instantly: a previously verified lease can remain valid for at most
30 seconds. Applications with a requirement for continuous duplicate enforcement
should use local mode.

## Approvals and credentials

By default, reload synchronization only commits policies. Review and deploy a
selected commit through the CLI/dashboard. To deploy future commits without a
separate approval, open the website application's **Deployment settings**, choose
the Cloudflare zone, and enable **Autopush**. Enabling it also schedules the latest
existing commit. This server setting is authoritative; legacy local auto-deploy
flags no longer approve changes. The website's scheduled `manage.py cloud_worker`
creates previews, validates, approves, executes, and verifies automatic deployments.
Until verification succeeds, all explicit blocks stay local. Uncertain/partial
deployments are never automatically resumed.

Use either OS-keyring login method for development or an expiring scoped `BOUVANE_TOKEN` for
services. The token needs `policies:read`, `policies:write`, and `deployments:read`;
manual CLI preview/approval additionally requires `deployments:write`. Server
Autopush runs under the user who enabled it, with current permission checks. Renew
expired tokens through the normal credential workflow. Tokens are reread during
refresh; invalid/expired credentials cause local fallback, not startup failure.

```powershell
bouvane edge          # inspect saved configuration
bouvane edge --local  # keep explicit blocks local and stop cloud synchronization
```

Reload after switching modes. Local mode retains the bridge and its explicit
blocks; it does not call the cloud. No production mode is enabled automatically.
