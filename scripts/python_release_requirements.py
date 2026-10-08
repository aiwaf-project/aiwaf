"""Print installed production dependencies, excluding release and audit tooling."""
from importlib.metadata import distribution

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    pending = [("aiwaf", ("django", "flask", "fastapi", "rust"))]
    seen, versions = set(), {}
    while pending:
        name, extras = pending.pop()
        key = (canonicalize_name(name), tuple(sorted(extras)))
        if key in seen:
            continue
        seen.add(key)
        installed = distribution(name)
        # The candidate itself may not exist in the registry yet.
        if key[0] != "aiwaf":
            versions[key[0]] = installed.version
        for raw in installed.requires or ():
            requirement = Requirement(raw)
            if not requirement.marker or any(
                requirement.marker.evaluate({"extra": extra})
                for extra in ("", *extras)
            ):
                pending.append((requirement.name, tuple(requirement.extras)))
    for name, version in sorted(versions.items()):
        print(f"{name}=={version}")


if __name__ == "__main__":
    main()
