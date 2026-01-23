from __future__ import annotations

import collections.abc as cabc
import contextlib
import inspect
import typing
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto


class ExpectionKind(Enum):
    FUNCTION = auto()
    COROUTINE = auto()
    CONTEXT_MANAGER = auto()
    ASYNC_CONTEXT_MANAGER = auto()


@dataclass(frozen=False, kw_only=True, slots=True)
class Expectation:
    method_name: str

    kind: ExpectionKind = ExpectionKind.FUNCTION

    args: tuple[object, ...] = field(default_factory=tuple)
    kwargs: dict[str, object] = field(default_factory=dict)

    returns: object | None = None
    raises: BaseException | None = AssertionError("Expectation was not properly initialized")

    def get_result(self) -> object:
        if self.raises is not None:
            raise self.raises

        return self.returns


class CoroutineExpectation(Expectation):
    kind: ExpectionKind = ExpectionKind.COROUTINE

    async def get_result(self) -> object:
        if self.raises is not None:
            raise self.raises

        return self.returns


class ContextManagerExpectation(Expectation):
    kind: ExpectionKind = ExpectionKind.CONTEXT_MANAGER

    @contextlib.contextmanager
    def get_result(self) -> cabc.Generator[object]:
        if self.raises is not None:
            raise self.raises

        yield self.returns


class AsyncContextManagerExpectation(Expectation):
    kind: ExpectionKind = ExpectionKind.ASYNC_CONTEXT_MANAGER

    @contextlib.asynccontextmanager
    async def get_result(self) -> cabc.AsyncGenerator[object]:
        if self.raises is not None:
            raise self.raises

        yield self.returns


class Registrar(typing.Protocol):
    def register(self, expectation: Expectation) -> None: ...


class ExpectationFinder(typing.Protocol):
    def pop_fifo_expectation(self, method_name: str) -> Expectation | None: ...


class Result:
    _expectation: Expectation
    _registrar: Registrar

    def __init__(self, expectation: Expectation, registrar: Registrar) -> None:
        self._expectation = expectation
        self._registrar = registrar


class Raises(Result):
    def raises(self, exception: BaseException) -> None:
        self._expectation.returns = None
        self._expectation.raises = exception

        self._registrar.register(self._expectation)


class Returns[T](Raises):
    def returns(self, value: T) -> None:
        self._expectation.returns = value
        self._expectation.raises = None

        self._registrar.register(self._expectation)


class Yields[T](Returns[T]):
    def yields(self, values: T) -> None:
        self._expectation.returns = values
        self._expectation.raises = None

        self._registrar.register(self._expectation)


class ArgsKwargs[**P, R]:
    def __init__(self, method_name: str, registrar: Registrar) -> None:
        self._method_name = method_name
        self._registrar = registrar

    def called_with(self, *args: P.args, **kwargs: P.kwargs) -> Returns[R]:
        expectation = Expectation(method_name=self._method_name, args=args, kwargs=kwargs)
        return Returns(expectation, self._registrar)


class CoroutineArgsKwargs[**P, R](ArgsKwargs[P, R]):
    def awaited_with(self, *args: P.args, **kwargs: P.kwargs) -> Returns[R]:
        expectation = CoroutineExpectation(method_name=self._method_name, args=args, kwargs=kwargs)
        return Returns(expectation, self._registrar)


class ContextManagerArgsKwargs[**P, R](ArgsKwargs[P, R]):
    def entered_with(self, *args: P.args, **kwargs: P.kwargs) -> Yields[R]:
        expectation = ContextManagerExpectation(method_name=self._method_name, args=args, kwargs=kwargs)
        return Yields(expectation, self._registrar)


class AsyncContextManagerArgsKwargs[**P, R](ArgsKwargs[P, R]):
    def async_entered_with(self, *args: P.args, **kwargs: P.kwargs) -> Yields[R]:
        expectation = AsyncContextManagerExpectation(method_name=self._method_name, args=args, kwargs=kwargs)
        return Yields(expectation, self._registrar)


class AnyArgsKwargs[**P, R](CoroutineArgsKwargs[P, R], ContextManagerArgsKwargs[P, R], AsyncContextManagerArgsKwargs[P, R]): ...


@typing.final
class MockController[T]:
    def __init__(self, target: type[T]) -> None:
        self._target = target
        self._expectations: defaultdict[str, list[Expectation]] = defaultdict(list)
        self._mock = typing.cast(T, _MockProxyImpl(target, self))

    def register(self, expectation: Expectation) -> None:
        self._expectations[expectation.method_name].append(expectation)

    def pop_fifo_expectation(self, method_name: str) -> Expectation | None:
        expectations = self._expectations.get(method_name, [])
        if not expectations:
            return None
        # FIFO ordering - consume first expectation
        return expectations.pop(0)

    @property
    def mock(self) -> T:
        return self._mock

    def verify(self) -> None:
        unsatisfied: list[str] = []
        for method_name, expectations in self._expectations.items():
            if expectations:
                unsatisfied.append(f"missing {len(expectations)} call(s) to '{method_name}'")

        if unsatisfied:
            msg = f"Unsatisfied expectations:\n{'\n'.join(unsatisfied)}"
            raise AssertionError(msg)

    def reset(self) -> None:
        self._expectations.clear()


@typing.final
class _MockProxyImpl[T]:
    def __init__(self, target: type[T], handler: ExpectationFinder) -> None:
        self._target = target
        self._handler = handler

    def __getattr__(self, name: str) -> object:
        if name.startswith("_") and not name.startswith("__"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def _mock_method(*args: object, **kwargs: object) -> object:
            expectation = self._handler.pop_fifo_expectation(name)

            if expectation is None:
                msg = f"Unexpected call to '{name}' with args={args}, kwargs={kwargs}. No expectation was set for this call."
                raise AssertionError(msg)

            self._assert_expectation_args(expectation, name, args, kwargs)
            return expectation.get_result()

        return _mock_method

    def _assert_expectation_args(
        self, expectation: Expectation, method_name: str, call_args: tuple[object, ...], call_kwargs: dict[str, object]
    ) -> None:
        want_kwargs = self._convert_args_to_kwargs(method_name, expectation.args)
        want_kwargs.update(expectation.kwargs)

        got_kwargs = self._convert_args_to_kwargs(method_name, call_args)
        got_kwargs.update(call_kwargs)

        diffs: list[str] = []

        for key, want_value in want_kwargs.items():
            if key not in got_kwargs:
                diffs.append(f"expected {key}={want_value!r}, but '{key}' is missing")
                continue

            got_value = got_kwargs[key]
            if got_value != want_value:
                diffs.append(f"expected {key}={want_value!r}, but got {key}={got_value!r}")

        for key in got_kwargs.keys():
            if key not in want_kwargs:
                diffs.append(f"unexpected {key}={got_kwargs[key]!r}")

        if diffs:
            raise AssertionError(f"Unexpected args for '{method_name}':\n{'\n'.join(diffs)}")

    def _convert_args_to_kwargs(self, method_name: str, args: tuple[object, ...]) -> dict[str, object]:
        target_method = getattr(self._target, method_name, None)
        if not target_method:
            raise AssertionError(f"Method '{method_name}' not found on target type {self._target}")

        if not args:
            return {}

        signature = inspect.signature(target_method)
        param_names = [p for p in signature.parameters if p != "self"]

        return dict(zip(param_names, args))


@typing.final
class Mockamorph[T]:
    def __init__(self, target: type[T]) -> None:
        self._target = target
        self._ctrl = MockController(target)

    def get_mock(self) -> T:
        return self._ctrl.mock

    @typing.overload
    def expect[**P, R](
        self,
        method: cabc.Callable[typing.Concatenate[T, P], typing.ContextManager[R]],
    ) -> ContextManagerArgsKwargs[P, R]: ...

    @typing.overload
    def expect[**P, R](
        self,
        method: cabc.Callable[typing.Concatenate[T, P], typing.AsyncContextManager[R]],
    ) -> AsyncContextManagerArgsKwargs[P, R]: ...

    @typing.overload
    def expect[**P, R](
        self,
        method: cabc.Callable[typing.Concatenate[T, P], cabc.Awaitable[R]],
    ) -> CoroutineArgsKwargs[P, R]: ...

    @typing.overload
    def expect[**P, R](
        self,
        method: cabc.Callable[typing.Concatenate[T, P], R],
    ) -> ArgsKwargs[P, R]: ...

    def expect[**P, R](
        self,
        method: cabc.Callable[typing.Concatenate[T, P], typing.ContextManager[R] | typing.AsyncContextManager[R] | cabc.Awaitable[R] | R],
    ) -> ArgsKwargs[P, R]:
        if method.__name__.startswith("_"):
            raise AttributeError("Cannot set expectations on private attribute")

        return AnyArgsKwargs[P, R](method.__name__, self._ctrl)

    def verify(self) -> None:
        self._ctrl.verify()

    def reset(self) -> None:
        self._ctrl.reset()

    def __enter__(self) -> Mockamorph[T]:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        _ = exc_type, exc_val, exc_tb
        self.verify()

    async def __aenter__(self) -> Mockamorph[T]:
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        _ = exc_type, exc_val, exc_tb
        self.verify()
