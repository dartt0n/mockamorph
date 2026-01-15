from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol

import pytest

from mockamorph import Mockamorph


def test_mock_protocol_returns_value() -> None:
    class SomeInterface(Protocol):
        def do_stuff(self) -> int: ...

    class SomeUsecase:
        def __init__(self, impl: SomeInterface) -> None:
            self.__impl = impl

        def business_logic(self) -> int:
            return self.__impl.do_stuff() * 5

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().do_stuff().called_with().returns(5)
        usecase = SomeUsecase(ctrl.get_mock())

        assert usecase.business_logic() == 25


def test_mock_abstract_class_returns_value() -> None:
    class SomeInterface(ABC):
        @abstractmethod
        def do_stuff(self) -> int: ...

    class SomeUsecase:
        def __init__(self, impl: SomeInterface) -> None:
            self.__impl = impl

        def business_logic(self) -> int:
            return self.__impl.do_stuff() * 5

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().do_stuff().called_with().returns(5)

        usecase = SomeUsecase(ctrl.get_mock())
        assert usecase.business_logic() == 25


def test_returns_tuple_unpacking() -> None:
    class SomeInterface(ABC):
        @abstractmethod
        def do_stuff(self) -> tuple[int, str, bool]: ...

    class SomeUsecase:
        def __init__(self, impl: SomeInterface) -> None:
            self.__impl = impl

        def business_logic(self) -> str:
            x, y, z = self.__impl.do_stuff()
            if z:
                return y * x

            return ""

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().do_stuff().called_with().returns(2, "Y", True)
        usecase = SomeUsecase(ctrl.get_mock())

        assert usecase.business_logic() == "YY"


def test_sequential_calls_return_fifo() -> None:
    class SomeInterface(Protocol):
        def do_stuff(self) -> int: ...

    class SomeUsecase:
        def __init__(self, impl: SomeInterface) -> None:
            self.__impl = impl

        def business_logic(self) -> int:
            return self.__impl.do_stuff() + 1

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().do_stuff().called_with().returns(1)
        ctrl.expect().do_stuff().called_with().returns(2)
        ctrl.expect().do_stuff().called_with().returns(3)

        usecase = SomeUsecase(ctrl.get_mock())

        assert usecase.business_logic() == 2
        assert usecase.business_logic() == 3
        assert usecase.business_logic() == 4


def test_unexpected_call_raises() -> None:
    class SomeInterface(Protocol):
        def do_stuff(self) -> int: ...

    class SomeUsecase:
        def __init__(self, impl: SomeInterface) -> None:
            self.__impl = impl

        def business_logic(self) -> int:
            return self.__impl.do_stuff() + 1

    with Mockamorph(SomeInterface) as ctrl:
        usecase = SomeUsecase(ctrl.get_mock())

        with pytest.raises(AssertionError):
            _ = usecase.business_logic()


def test_called_with_args() -> None:
    class SomeInterface(Protocol):
        def do_stuff(self, x: int) -> str: ...

    class SomeUsecase:
        def __init__(self, impl: SomeInterface) -> None:
            self.__impl = impl

        def business_logic(self, value: int) -> str:
            return self.__impl.do_stuff(value) + "!"

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().do_stuff().called_with(42).returns("hello")
        usecase = SomeUsecase(ctrl.get_mock())

        assert usecase.business_logic(42) == "hello!"


def test_verify_fails_on_unsatisfied() -> None:
    class SomeInterface(Protocol):
        def do_stuff(self) -> int: ...

    with pytest.raises(AssertionError, match="Unsatisfied expectations"):
        with Mockamorph(SomeInterface) as ctrl:
            ctrl.expect().do_stuff().called_with().returns(5)


def test_reset_clears_all() -> None:
    class SomeInterface(Protocol):
        def do_stuff(self) -> int: ...

    mock = Mockamorph(SomeInterface)
    mock.expect().do_stuff().called_with().returns(5)
    mock.reset()
    mock.verify()


def test_raises_configured_exception() -> None:
    class SomeInterface(Protocol):
        def do_stuff(self) -> int: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().do_stuff().called_with().raises(ValueError("test error"))

        with pytest.raises(ValueError, match="test error"):
            _ = ctrl.get_mock().do_stuff()


def test_same_method_returns_or_raises() -> None:
    class SomeInterface(Protocol):
        def do_stuff(self, value: int) -> int: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().do_stuff().called_with(1).returns(10)
        ctrl.expect().do_stuff().called_with(2).raises(ValueError("test error"))
        mock = ctrl.get_mock()

        assert mock.do_stuff(1) == 10

        with pytest.raises(ValueError, match="test error"):
            _ = mock.do_stuff(2)


def test_args_kwargs_matching() -> None:
    class SomeInterface(Protocol):
        def do_stuff(self, a: int, b: str) -> int: ...

    class SomeUsecase:
        def __init__(self, mock: SomeInterface):
            self.mock = mock

        def use(self, *args: Any, **kwargs: Any) -> None:
            self.mock.do_stuff(*args, **kwargs)

    with Mockamorph(SomeInterface) as ctrl:
        # keyword only
        ctrl.expect().do_stuff().called_with(a=1, b="test1").returns(5)
        # one arg, one kwargs
        ctrl.expect().do_stuff().called_with(2, b="test2").returns(5)
        # args
        ctrl.expect().do_stuff().called_with(3, "test3").returns(5)

        # exptected with different style of args
        SomeUsecase(ctrl.get_mock()).use(1, "test1")
        SomeUsecase(ctrl.get_mock()).use(2, "test2")
        SomeUsecase(ctrl.get_mock()).use(a=3, b="test3")

        with pytest.raises(AssertionError, match="Unexpected call"):
            SomeUsecase(ctrl.get_mock()).use(a=10, b="test3")


def test_multiple_methods_on_mock() -> None:
    class SomeInterface(Protocol):
        def method_a(self, x: int) -> str: ...
        def method_b(self, y: str) -> int: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().method_a().called_with(1).returns("one")
        ctrl.expect().method_b().called_with("two").returns(2)

        mock = ctrl.get_mock()
        assert mock.method_a(1) == "one"
        assert mock.method_b("two") == 2


def test_method_with_different_args() -> None:
    class SomeInterface(Protocol):
        def calculate(self, x: int, y: int) -> int: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().calculate().called_with(1, 2).returns(3)
        ctrl.expect().calculate().called_with(10, 20).returns(30)

        mock = ctrl.get_mock()
        assert mock.calculate(1, 2) == 3
        assert mock.calculate(10, 20) == 30


def test_returns_none() -> None:
    class SomeInterface(Protocol):
        def do_nothing(self) -> None: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().do_nothing().called_with().returns(None)

        mock = ctrl.get_mock()
        result = mock.do_nothing()  # type: ignore[func-returns-value]
        assert result is None


def test_returns_dict() -> None:
    class SomeInterface(Protocol):
        def get_data(self) -> dict[str, list[int]]: ...

    expected_data = {"numbers": [1, 2, 3], "more": [4, 5]}

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().get_data().called_with().returns(expected_data)

        mock = ctrl.get_mock()
        assert mock.get_data() == expected_data


def test_raises_custom_exception_with_attrs() -> None:
    class CustomError(Exception):
        def __init__(self, code: int, message: str):
            self.code = code
            self.message = message
            super().__init__(message)

    class SomeInterface(Protocol):
        def risky_operation(self) -> int: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().risky_operation().called_with().raises(
            CustomError(404, "not found")
        )

        mock = ctrl.get_mock()
        with pytest.raises(CustomError) as exc_info:
            mock.risky_operation()

        assert exc_info.value.code == 404
        assert exc_info.value.message == "not found"


def test_interleaved_method_calls() -> None:
    class SomeInterface(Protocol):
        def foo(self) -> int: ...
        def bar(self) -> str: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().foo().called_with().returns(1)
        ctrl.expect().bar().called_with().returns("a")
        ctrl.expect().foo().called_with().returns(2)
        ctrl.expect().bar().called_with().returns("b")

        mock = ctrl.get_mock()
        assert mock.foo() == 1
        assert mock.bar() == "a"
        assert mock.foo() == 2
        assert mock.bar() == "b"


def test_verify_passes_when_satisfied() -> None:
    class SomeInterface(Protocol):
        def do_stuff(self) -> int: ...

    ctrl = Mockamorph(SomeInterface)
    ctrl.expect().do_stuff().called_with().returns(5)

    mock = ctrl.get_mock()
    assert mock.do_stuff() == 5

    ctrl.verify()


def test_reset_then_new_expectations() -> None:
    class SomeInterface(Protocol):
        def do_stuff(self) -> int: ...

    ctrl = Mockamorph(SomeInterface)
    ctrl.expect().do_stuff().called_with().returns(5)
    ctrl.reset()

    ctrl.expect().do_stuff().called_with().returns(10)
    mock = ctrl.get_mock()
    assert mock.do_stuff() == 10

    ctrl.verify()


def test_kwargs_only_args() -> None:
    class SomeInterface(Protocol):
        def configure(self, *, host: str, port: int) -> str: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().configure().called_with(host="localhost", port=8080).returns("ok")

        mock = ctrl.get_mock()
        assert mock.configure(host="localhost", port=8080) == "ok"


def test_alternating_returns_and_raises() -> None:
    class SomeInterface(Protocol):
        def process(self, item: str) -> str: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().process().called_with("good").returns("processed")
        ctrl.expect().process().called_with("bad").raises(ValueError("invalid item"))
        ctrl.expect().process().called_with("good2").returns("processed2")

        mock = ctrl.get_mock()
        assert mock.process("good") == "processed"

        with pytest.raises(ValueError, match="invalid item"):
            mock.process("bad")

        assert mock.process("good2") == "processed2"


def test_abstract_class_multiple_methods() -> None:
    class SomeInterface(ABC):
        @abstractmethod
        def read(self, key: str) -> str: ...

        @abstractmethod
        def write(self, key: str, value: str) -> bool: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().read().called_with("mykey").returns("myvalue")
        ctrl.expect().write().called_with("newkey", "newvalue").returns(True)

        mock = ctrl.get_mock()
        assert mock.read("mykey") == "myvalue"
        assert mock.write("newkey", "newvalue") is True


def test_fifo_ordering() -> None:
    class SomeInterface(Protocol):
        def step(self) -> int: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().step().called_with().returns(1)
        ctrl.expect().step().called_with().returns(2)
        ctrl.expect().step().called_with().returns(3)

        mock = ctrl.get_mock()
        assert mock.step() == 1
        assert mock.step() == 2
        assert mock.step() == 3


def test_private_attr_expectation_rejected() -> None:
    class SomeInterface(Protocol): ...

    with Mockamorph(SomeInterface) as ctrl:
        with pytest.raises(AttributeError, match="private attribute"):
            ctrl.expect()._private_method()


def test_returns_callable_value() -> None:
    class SomeInterface(Protocol):
        def get_callback(self) -> Callable[[int], int]: ...

    def my_callback(x: int) -> int:
        return x * 2

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().get_callback().called_with().returns(my_callback)

        mock = ctrl.get_mock()
        callback = mock.get_callback()
        assert callback(5) == 10


def test_verify_lists_all_unsatisfied() -> None:
    class SomeInterface(Protocol):
        def foo(self) -> int: ...
        def bar(self) -> str: ...

    ctrl = Mockamorph(SomeInterface)
    ctrl.expect().foo().called_with().returns(1)
    ctrl.expect().foo().called_with().returns(2)
    ctrl.expect().bar().called_with().returns("a")

    with pytest.raises(AssertionError, match="Unsatisfied expectations"):
        ctrl.verify()


def test_no_args_method() -> None:
    class SomeInterface(Protocol):
        def no_args(self) -> str: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().no_args().called_with().returns("result")

        mock = ctrl.get_mock()
        assert mock.no_args() == "result"


def test_returns_bool() -> None:
    class SomeInterface(Protocol):
        def is_valid(self, value: int) -> bool: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().is_valid().called_with(1).returns(True)
        ctrl.expect().is_valid().called_with(0).returns(False)

        mock = ctrl.get_mock()
        assert mock.is_valid(1) is True
        assert mock.is_valid(0) is False


def test_returns_list() -> None:
    class SomeInterface(Protocol):
        def get_items(self) -> list[str]: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().get_items().called_with().returns(["a", "b", "c"])

        mock = ctrl.get_mock()
        assert mock.get_items() == ["a", "b", "c"]


def test_retry_pattern_raises_then_returns() -> None:
    class SomeInterface(Protocol):
        def retry_operation(self) -> str: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().retry_operation().called_with().raises(ConnectionError("failed"))
        ctrl.expect().retry_operation().called_with().returns("success")

        mock = ctrl.get_mock()

        with pytest.raises(ConnectionError):
            mock.retry_operation()

        assert mock.retry_operation() == "success"


class TextFixtureMockamorpthInterface(Protocol):
    def do_stuff(self) -> str: ...


@pytest.fixture
def mockamorph_fixture() -> Mockamorph[TextFixtureMockamorpthInterface]:
    ctrl = Mockamorph(TextFixtureMockamorpthInterface)
    ctrl.expect().do_stuff().called_with().returns("Hello world")
    return ctrl


@pytest.fixture
def mockamorpth_fixture_mock(
    mockamorph_fixture: Mockamorph[TextFixtureMockamorpthInterface],
) -> TextFixtureMockamorpthInterface:
    return mockamorph_fixture.get_mock()


def test_mockamorph_as_fixture_success(
    mockamorph_fixture: Mockamorph[TextFixtureMockamorpthInterface],
    mockamorpth_fixture_mock: TextFixtureMockamorpthInterface,
) -> None:
    # call is present
    assert mockamorpth_fixture_mock.do_stuff() == "Hello world"
    mockamorph_fixture.verify()


def test_mockamorph_as_fixture_failure(
    mockamorph_fixture: Mockamorph[TextFixtureMockamorpthInterface],
) -> None:
    # because no call happened
    with pytest.raises(AssertionError, match="Unsatisfied expectations"):
        mockamorph_fixture.verify()


@pytest.mark.asyncio
async def test_async_method() -> None:
    class AsyncInterface(Protocol):
        async def async_method(self) -> str: ...

    async with Mockamorph(AsyncInterface) as ctrl:
        ctrl.expect().async_method().awaited_with().returns("Hello world")

        mock = ctrl.get_mock()

        assert await mock.async_method() == "Hello world"


@pytest.mark.asyncio
async def test_async_method_raises() -> None:
    class AsyncInterface(Protocol):
        async def async_method(self) -> str: ...

    async with Mockamorph(AsyncInterface) as ctrl:
        ctrl.expect().async_method().awaited_with().raises(RuntimeError("Failed"))

        mock = ctrl.get_mock()

        with pytest.raises(RuntimeError, match="Failed"):
            await mock.async_method()
