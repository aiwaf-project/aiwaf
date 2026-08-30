# Contributing to AIWAF

AIWAF is a monorepo containing independently published Python and JavaScript packages.

## Repository layout

- `py/aiwaf/`: Python package source (imported as `aiwaf`)
- `js/`: JavaScript package source (published as `aiwaf-js`)
- `tests/`: Python tests
- `examples/`: Python examples and sandbox tooling
- `.github/workflows/`: package-specific CI and publishing

## Python development

Create a virtual environment and install the package with its framework adapters:

```bash
python -m venv .venv
python -m pip install -e ".[django,flask,fastapi]"
python -m pip install pytest httpx
```

Run the test groups independently so Django's global application state does not leak into the other adapters:

```bash
python manage.py test tests.django
python -m pytest tests/core tests/flask tests/fast tests/integration -q
```

Python package metadata lives in `pyproject.toml`. Keep the compatibility metadata in `setup.py` synchronized when changing dependencies or versions.

## JavaScript development

```bash
cd js
npm ci
npm test
npm pack --dry-run
```

From the repository root, the equivalent test command is `npm run test:js`.

## Releases

- Push a `python-v*` tag to build and publish `aiwaf` to PyPI.
- Push a `js-v*` tag to test and publish `aiwaf-js` to npm.

The workflows can also be run manually. PyPI uses trusted publishing; npm requires the `NPM_TOKEN` repository secret.
