import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
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

        with pytest.raises(AssertionError, match="Unexpected call to 'do_stuff'"):
            SomeUsecase(ctrl.get_mock()).use(a=10, b="test3")


def test_incorrect_args() -> None:
    class SomeInterface(Protocol):
        def method_a(self, x: int, y: str) -> str: ...

    with Mockamorph(SomeInterface) as ctrl:
        # expected with non-keyword args
        ctrl.expect().method_a().called_with(1, "test1").returns("one")

        with pytest.raises(
            AssertionError,
            match=re.escape(
                "Unexpected args for 'method_a':\nexpected x=1, but got x=2\nexpected y='test1', but got y='not-test'"
            ),
        ):
            # call with keyword args with wrong values
            ctrl.get_mock().method_a(x=2, y="not-test")


def test_method_with_nonmatching_args_no_sideeffect() -> None:
    class SomeInterface(Protocol):
        def method_a(self, x: int, y: str) -> str: ...

    with Mockamorph(SomeInterface) as ctrl:
        ctrl.expect().method_a().called_with(x=1, y="test1").raises(RuntimeError("!!!"))

        mock = ctrl.get_mock()
        with pytest.raises(AssertionError, match="Unexpected args for 'method_a'"):
            mock.method_a(2, y="test2")  # RuntimeError is NOT raised


def test_slash_in_argument_list() -> None:
    class SomeInterface(Protocol):
        def method_a(self, x: int, /, y: str) -> str: ...

    with Mockamorph(SomeInterface) as ctrl:
        # note: this does not follow original signature
        ctrl.expect().method_a().called_with(x=1, y="test1").returns("one")
        ctrl.expect().method_a().called_with(x=3, y="test3").returns("one")

        mock = ctrl.get_mock()
        with pytest.raises(AssertionError, match="Unexpected args for 'method_a'"):
            mock.method_a(2, y="test2")

        assert mock.method_a(3, y="test3") == "one"


def test_star_in_argument_list() -> None:
    class SomeInterface(Protocol):
        def method_a(self, x: int, *, y: str) -> str: ...

    with Mockamorph(SomeInterface) as ctrl:
        # note: this does not follow original signature
        ctrl.expect().method_a().called_with(1, "test1").returns("one")
        ctrl.expect().method_a().called_with(3, "test3").returns("one")

        mock = ctrl.get_mock()
        with pytest.raises(AssertionError, match="Unexpected args for 'method_a'"):
            mock.method_a(x=2, y="test2")

        assert mock.method_a(x=3, y="test3") == "one"


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


def test_private_attr_expectation_rejected() -> None:
    class SomeInterface(Protocol): ...

    with Mockamorph(SomeInterface) as ctrl:
        with pytest.raises(
            AttributeError, match="Cannot set expectations on private attribute"
        ):
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

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Unsatisfied expectations:\nmissing 2 call(s) to 'foo'\nmissing 1 call(s) to 'bar'"
        ),
    ):
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


def test_call_method_with_no_expectations() -> None:
    class Service(Protocol):
        def process(self) -> str: ...

    with pytest.raises(AssertionError, match="Unexpected call"):
        with Mockamorph(Service) as mock:
            mock.get_mock().process()


def test_call_method_more_times_than_expected() -> None:
    class Service(Protocol):
        def get_value(self) -> int: ...

    with Mockamorph(Service) as mock:
        mock.expect().get_value().called_with().returns(1)
        mock.expect().get_value().called_with().returns(2)

        m = mock.get_mock()
        assert m.get_value() == 1
        assert m.get_value() == 2
        with pytest.raises(AssertionError, match="Unexpected call to 'get_value'"):
            m.get_value()  # Third call - should fail


def test_call_wrong_method() -> None:
    class Service(Protocol):
        def method_a(self) -> str: ...
        def method_b(self) -> str: ...

    mock = Mockamorph(Service)
    mock.expect().method_a().called_with().returns("a")
    m = mock.get_mock()

    with pytest.raises(AssertionError, match="Unexpected call to 'method_b'"):
        m.method_b()  # Wrong method

    assert m.method_a() == "a"  # method_b did not consume expected call for method_a


def test_wrong_positional_arg_value() -> None:
    class Calculator(Protocol):
        def add(self, a: int, b: int) -> int: ...

    with Mockamorph(Calculator) as mock:
        mock.expect().add().called_with(1, 2).returns(3)

        with pytest.raises(
            AssertionError,
            match=re.escape("Unexpected args for 'add':\nexpected b=2, but got b=999"),
        ):
            mock.get_mock().add(1, 999)  # Wrong second arg


def test_wrong_keyword_arg_value() -> None:
    class Config(Protocol):
        def set_option(self, *, name: str, value: int) -> None: ...

    with Mockamorph(Config) as mock:
        mock.expect().set_option().called_with(name="debug", value=1).returns(None)

        with pytest.raises(
            AssertionError,
            match=re.escape(
                "Unexpected args for 'set_option':\nexpected value=1, but got value=0"
            ),
        ):
            mock.get_mock().set_option(name="debug", value=0)


def test_missing_required_arg() -> None:
    class Service(Protocol):
        def fetch(self, key: str) -> str: ...

    with Mockamorph(Service) as mock:
        mock.expect().fetch().called_with("mykey").returns("value")

        with pytest.raises(
            AssertionError,
            match=re.escape(
                "Unexpected args for 'fetch':\nexpected key='mykey', but 'key' is missing"
            ),
        ):
            #  missing key parameter
            mock.get_mock().fetch()  # pyright: ignore[reportCallIssue]  # ty:ignore[missing-argument]


def test_provided_extra_kwargs() -> None:
    class SomeInterface(Protocol):
        def method_a(self, a: str, b: int) -> None: ...

    with Mockamorph(SomeInterface) as mock:
        mock.expect().method_a().called_with(a="test", b=5).returns(None)

        with pytest.raises(AssertionError, match="unexpected extra=True"):
            # extra argument
            mock.get_mock().method_a(a="test", b=5, extra=True)  # ty:ignore[unknown-argument]  # pyright: ignore[reportCallIssue]


def test_verify_with_partial_satisfaction() -> None:
    class Service(Protocol):
        def step1(self) -> None: ...
        def step2(self) -> None: ...
        def step3(self) -> None: ...

    ctrl = Mockamorph(Service)
    ctrl.expect().step1().called_with().returns(None)
    ctrl.expect().step2().called_with().returns(None)
    ctrl.expect().step3().called_with().returns(None)

    m = ctrl.get_mock()
    m.step1()
    m.step2()
    # step3 not called

    with pytest.raises(
        AssertionError,
        match=re.escape("Unsatisfied expectations:\nmissing 1 call(s) to 'step3"),
    ):
        ctrl.verify()


def test_verify_called_multiple_times() -> None:
    class Service(Protocol):
        def action(self) -> str: ...

    ctrl = Mockamorph(Service)
    ctrl.expect().action().called_with().returns("done")

    ctrl.get_mock().action()

    # Multiple verify calls should all pass
    ctrl.verify()
    ctrl.verify()
    ctrl.verify()


def test_verify_counts_remaining_expectations() -> None:
    class Service(Protocol):
        def method(self) -> int: ...

    ctrl = Mockamorph(Service)
    for i in range(5):
        ctrl.expect().method().called_with().returns(i)

    ctrl.get_mock().method()  # Only consume 1 of 5

    with pytest.raises(
        AssertionError,
        match=re.escape("Unsatisfied expectations:\nmissing 4 call(s) to 'method'"),
    ):
        ctrl.verify()


def test_mock_private_attr_access_rejected() -> None:
    class Service(Protocol):
        def method(self) -> str: ...

    ctrl = Mockamorph(Service)
    with pytest.raises(AttributeError):
        ctrl.get_mock()._something  # pyright: ignore[reportAttributeAccessIssue]  # ty:ignore[unresolved-attribute]


def test_reuse_after_context_exit() -> None:
    class Service(Protocol):
        def get(self) -> int: ...

    ctrl = Mockamorph(Service)

    with ctrl:
        ctrl.expect().get().called_with().returns(1)
        assert ctrl.get_mock().get() == 1

    # After exit, set new expectations
    ctrl.expect().get().called_with().returns(2)
    assert ctrl.get_mock().get() == 2
    ctrl.verify()


@pytest.mark.asyncio
async def test_async_method_not_awaited_returns_future() -> None:
    class AsyncService(Protocol):
        async def fetch(self) -> str: ...

    async with Mockamorph(AsyncService) as mock:
        mock.expect().fetch().awaited_with().returns("data")

        result = mock.get_mock().fetch()
        assert asyncio.isfuture(result) or asyncio.iscoroutine(result)
        # Clean up by awaiting
        assert await result == "data"


@pytest.mark.asyncio
async def test_async_multiple_awaits_fifo() -> None:
    class AsyncService(Protocol):
        async def fetch(self) -> int: ...

    async with Mockamorph(AsyncService) as mock:
        mock.expect().fetch().awaited_with().returns(1)
        mock.expect().fetch().awaited_with().returns(2)
        mock.expect().fetch().awaited_with().returns(3)

        m = mock.get_mock()
        assert await m.fetch() == 1
        assert await m.fetch() == 2
        assert await m.fetch() == 3


@pytest.mark.asyncio
async def test_async_raises_and_returns_interleaved() -> None:
    class AsyncService(Protocol):
        async def call_api(self) -> str: ...

    async with Mockamorph(AsyncService) as mock:
        mock.expect().call_api().awaited_with().raises(ConnectionError("timeout"))
        mock.expect().call_api().awaited_with().returns("success")
        mock.expect().call_api().awaited_with().raises(ConnectionError("again"))

        m = mock.get_mock()

        with pytest.raises(ConnectionError):
            await m.call_api()

        assert await m.call_api() == "success"

        with pytest.raises(ConnectionError):
            await m.call_api()


@pytest.mark.asyncio
async def test_sync_expectation_for_async_method_works() -> None:
    class AsyncService(Protocol):
        async def fetch(self) -> str: ...

    class AsyncUsecase:
        async def use(self, service: AsyncService) -> str:
            return await service.fetch()

    # Using called_with instead of awaited_with
    async with Mockamorph(AsyncService) as mock:
        mock.expect().fetch().called_with().returns("some-value")

        with pytest.raises(
            TypeError, match="object str can't be used in 'await' expression"
        ):
            await AsyncUsecase().use(mock.get_mock())


def test_call_nonexistent_method() -> None:
    class Service(Protocol):
        def existing_method(self) -> str: ...

    with pytest.raises(AssertionError, match="Unexpected call"):
        with Mockamorph(Service) as mock:
            # No expectation set, so it fails
            mock.get_mock().nonexistent_method()  # type: ignore[attr-defined]


def test_expect_nonexistent_method_then_call() -> None:
    class Service(Protocol):
        def real_method(self) -> str: ...

    with Mockamorph(Service) as mock:
        mock.expect().fake_method().called_with().returns("faked")
        with pytest.raises(
            AssertionError, match="Method 'fake_method' not found on target type"
        ):
            mock.get_mock().fake_method()  # type: ignore[attr-defined]


def test_empty_protocol() -> None:
    class EmptyService(Protocol):
        pass

    with Mockamorph(EmptyService) as mock:
        _ = mock.get_mock()


def test_empty_abc() -> None:
    class EmptyAbstract(ABC):
        pass

    with Mockamorph(EmptyAbstract) as mock:
        _ = mock.get_mock()


def test_get_mock_returns_same_instance() -> None:
    class Service(Protocol):
        def method(self) -> str: ...

    ctrl = Mockamorph(Service)
    mock1 = ctrl.get_mock()
    mock2 = ctrl.get_mock()

    assert mock1 is mock2


def test_expectations_set_after_get_mock() -> None:
    class Service(Protocol):
        def method(self) -> int: ...

    ctrl = Mockamorph(Service)
    mock = ctrl.get_mock()

    # Set expectation after getting mock
    ctrl.expect().method().called_with().returns(42)

    assert mock.method() == 42
    ctrl.verify()


def test_none_as_argument() -> None:
    class Service(Protocol):
        def process(self, value: str | None) -> str: ...

    with Mockamorph(Service) as mock:
        mock.expect().process().called_with(None).returns("null")

        assert mock.get_mock().process(None) == "null"


def test_nested_dict_argument() -> None:
    class Service(Protocol):
        def configure(self, config: dict[str, Any]) -> bool: ...

    config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "credentials": {"user": "admin", "password": "secret"},
        },
        "cache": {"enabled": True, "ttl": 300},
    }

    with Mockamorph(Service) as mock:
        mock.expect().configure().called_with(config).returns(True)

        assert mock.get_mock().configure(config) is True


def test_list_argument() -> None:
    class Service(Protocol):
        def batch_process(self, items: list[int]) -> int: ...

    with Mockamorph(Service) as mock:
        mock.expect().batch_process().called_with([1, 2, 3, 4, 5]).returns(15)

        assert mock.get_mock().batch_process([1, 2, 3, 4, 5]) == 15


def test_dataclass_argument() -> None:
    @dataclass
    class Request:
        id: int
        payload: str
        metadata: dict[str, str] = field(default_factory=dict)

    class Service(Protocol):
        def handle(self, request: Request) -> str: ...

    req = Request(id=1, payload="data", metadata={"trace": "abc123"})

    with Mockamorph(Service) as mock:
        mock.expect().handle().called_with(req).returns("ok")

        assert mock.get_mock().handle(req) == "ok"


def test_callable_argument() -> None:
    class Service(Protocol):
        def register_callback(self, callback: Any) -> None: ...

    def my_callback() -> None:
        pass

    with Mockamorph(Service) as mock:
        mock.expect().register_callback().called_with(my_callback).returns(None)

        mock.get_mock().register_callback(my_callback)


def test_return_exception_object_not_raise() -> None:
    class Service(Protocol):
        def get_error(self) -> Exception: ...

    error = ValueError("this is returned, not raised")

    with Mockamorph(Service) as mock:
        mock.expect().get_error().called_with().returns(error)

        result = mock.get_mock().get_error()
        assert result is error
        assert isinstance(result, ValueError)


def test_return_mock_object() -> None:
    class InnerService(Protocol):
        def inner_method(self) -> str: ...

    class OuterService(Protocol):
        def get_inner(self) -> InnerService: ...

    with (
        Mockamorph(InnerService) as inner_mock,
        Mockamorph(OuterService) as outer_mock,
    ):
        inner_mock.expect().inner_method().called_with().returns("inner result")
        outer_mock.expect().get_inner().called_with().returns(inner_mock.get_mock())

        outer = outer_mock.get_mock()
        inner = outer.get_inner()
        assert inner.inner_method() == "inner result"


def test_return_generator_function() -> None:
    class Service(Protocol):
        def get_generator(self) -> Any: ...

    def my_gen() -> Any:
        yield 1
        yield 2
        yield 3

    with Mockamorph(Service) as mock:
        mock.expect().get_generator().called_with().returns(my_gen)

        result = mock.get_mock().get_generator()
        assert list(result()) == [1, 2, 3]


def test_return_tuple_single_element() -> None:
    class Service(Protocol):
        def get_single(self) -> tuple[int]: ...
        def get_value(self) -> int: ...

    with Mockamorph(Service) as mock:
        mock.expect().get_single().called_with().returns((42,))
        assert mock.get_mock().get_single() == (42,)

        mock.expect().get_value().called_with().returns(42)
        assert mock.get_mock().get_value() == 42


def test_raise_base_exception() -> None:
    class Service(Protocol):
        def critical(self) -> None: ...

    with Mockamorph(Service) as mock:
        mock.expect().critical().called_with().raises(SystemExit(1))

        with pytest.raises(SystemExit):
            mock.get_mock().critical()


def test_raise_exception_with_cause() -> None:
    class Service(Protocol):
        def risky(self) -> None: ...

    original = ValueError("original error")
    chained = RuntimeError("wrapper error")
    chained.__cause__ = original

    with Mockamorph(Service) as mock:
        mock.expect().risky().called_with().raises(chained)

        with pytest.raises(RuntimeError) as exc_info:
            mock.get_mock().risky()

        assert exc_info.value.__cause__ is original
