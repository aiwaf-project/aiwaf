from aiwaf.flask.decorators import aiwaf_exempt


def test_exempt_decorator_marks_and_executes_function():
    @aiwaf_exempt
    def endpoint(value):
        return value

    assert endpoint("ok") == "ok"
    assert endpoint._aiwaf_exempt is True
