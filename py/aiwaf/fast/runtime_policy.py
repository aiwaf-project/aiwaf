"""App-owned route facts and trained models shared by both FastAPI runtimes."""
import logging
import re

from aiwaf.core.model_serialization import load_model_artifact
from aiwaf.core.rust_backend import rust_isolation_forest_from_json


def _validate_model_state(state):
    # Older extension builds assume these keys exist when decoding a forest.
    required = {"nEstimators", "bootstrap", "verbose", "warmStart", "maxSamples_",
                "maxFeatures_", "offset_", "nFeaturesIn_", "trees", "estimatorsFeatures"}
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError("incomplete model state")
    if state["nFeaturesIn_"] != 6 or not isinstance(state["trees"], list) or not state["trees"]:
        raise ValueError("model must be fitted on six request features")
    pending = list(state["trees"])
    while pending:
        tree = pending.pop()
        if not isinstance(tree, dict) or not {"depth", "maxDepth", "splitAttr", "splitValue", "size"} <= tree.keys():
            raise ValueError("incomplete model tree")
        pending.extend(tree[key] for key in ("left", "right") if tree.get(key) is not None)


def route_keywords(app):
    keywords, prefixes = set(), set()
    # Direct app.add_middleware(..., config="from_settings") receives the inner
    # middleware chain rather than the FastAPI instance.
    seen = set()
    while app is not None and not hasattr(app, "routes") and id(app) not in seen:
        seen.add(id(app))
        app = getattr(app, "app", None)

    def visit(routes, parent=""):
        for route in routes:
            path = parent + str(getattr(route, "path", "") or "")
            keywords.update(re.findall(r"([a-zA-Z]\w{2,})", path.lower()))
            first = path.strip("/").split("/")[0]
            if first and "{" not in first:
                prefixes.add(first.lower())
            visit(getattr(route, "routes", ()) or (), path)

    visit(getattr(app, "routes", ()) or ())
    return keywords, prefixes


def load_trained_model(config):
    """Load the JSON artifact written by FastAPI's trainer; invalid files fail open."""
    if config is None:
        return None
    path = config.get("ai_anomaly.model_path") or config.get("AIWAF_MODEL_PATH")
    if not path:
        return None
    try:
        artifact = load_model_artifact(path)
        if not isinstance(artifact, dict) or artifact.get("model_type") != "aiwaf_rust.IsolationForest":
            raise ValueError("expected an aiwaf-rust model artifact")
        state = artifact.get("model_state")
        _validate_model_state(state)
        model = rust_isolation_forest_from_json(state)
        if model is None:
            raise ValueError("model state could not be loaded")
        return model
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        logging.getLogger(__name__).warning("Could not load FastAPI AI model from %s: %s", path, error)
        return None
