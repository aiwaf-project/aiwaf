from types import SimpleNamespace

from aiwaf.core.uuid_tamper import collect_uuid_model_fields, uuid_exists_in_model_fields


class UUIDField:
    def __init__(self, name, *, unique=False):
        self.name = name
        self.unique = unique


class Manager:
    def filter(self, **query):
        return SimpleNamespace(exists=lambda: next(iter(query.values())) == "found")


def test_collect_and_query_uuid_model_fields():
    pk = UUIDField("id")
    unique = UUIDField("external_id", unique=True)
    ignored = UUIDField("other")
    model = SimpleNamespace(_meta=SimpleNamespace(pk=pk, fields=[pk, unique, ignored]), objects=Manager())
    fields = collect_uuid_model_fields([model], UUIDField)
    assert fields == [(model, "pk"), (model, "external_id")]
    assert uuid_exists_in_model_fields("found", fields)
    assert not uuid_exists_in_model_fields("missing", fields)
