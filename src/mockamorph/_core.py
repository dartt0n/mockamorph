from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Concatenate,
    Never,
    Protocol,
    cast,
    final,
)


class ExpectionKind(Enum):
    FUNCTION = auto()
    COROUTINE = auto()
    CONTEXT_MANAGER = auto()
    ASYNC_CONTEXT_MANAGER = auto()


@dataclass(frozen=False, kw_only=True, slots=True)
class Expectation:
    method_name: str

    kind: ExpectionKind = ExpectionKind.FUNCTION

    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)

    return_value: Any = None
    exception: BaseException | None = AssertionError(
        "Expectation was not properly initialized"
    )


class Registrar(Protocol):
    def register(self, expectation: Expectation) -> None: ...


class ExpectationFinder(Protocol):
    def find_expectation(self, method_name: str) -> Expectation | None: ...


@final
class ReturnSetter[T]:
    def __init__(self, expectation: Expectation, registrar: Registrar) -> None:
        self._expectation = expectation
        self._registrar = registrar

    def returns(self, value: T) -> None:
        self._expectation.return_value = value
        self._expectation.exception = None

        self._registrar.register(self._expectation)

    def raises(self, exception: BaseException) -> None:
        self._expectation.return_value = None
        self._expectation.exception = exception

        self._registrar.register(self._expectation)

    def yields(self, values: T) -> None:
        self._expectation.return_value = values
        self._expectation.exception = None

        self._registrar.register(self._expectation)


@final
class CallArgsSetter[**ParamT, ReturnT]:
    def __init__(self, expectation: Expectation, registrar: Registrar) -> None:
        self._expectation = expectation
        self._registrar = registrar

    def called_with(
        self, *args: ParamT.args, **kwargs: ParamT.kwargs
    ) -> ReturnSetter[ReturnT]:
        self._expectation.args = args
        self._expectation.kwargs = kwargs
        self._expectation.kind = ExpectionKind.FUNCTION
        return ReturnSetter(self._expectation, self._registrar)

    def awaited_with(
        self, *args: ParamT.args, **kwargs: ParamT.kwargs
    ) -> ReturnSetter[ReturnT]:
        self._expectation.args = args
        self._expectation.kwargs = kwargs
        self._expectation.kind = ExpectionKind.COROUTINE
        return ReturnSetter(self._expectation, self._registrar)

    def entered_with(
        self, *args: ParamT.args, **kwargs: ParamT.kwargs
    ) -> ReturnSetter[ReturnT]:
        self._expectation.args = args
        self._expectation.kwargs = kwargs
        self._expectation.kind = ExpectionKind.CONTEXT_MANAGER
        return ReturnSetter(self._expectation, self._registrar)

    def async_entered_with(
        self, *args: ParamT.args, **kwargs: ParamT.kwargs
    ) -> ReturnSetter[ReturnT]:
        self._expectation.args = args
        self._expectation.kwargs = kwargs
        self._expectation.kind = ExpectionKind.ASYNC_CONTEXT_MANAGER
        return ReturnSetter(self._expectation, self._registrar)


@final
class MockController[T]:
    def __init__(self, target: type[T]) -> None:
        self._target = target
        self._expectations: defaultdict[str, list[Expectation]] = defaultdict(list)
        self._mock = cast(T, _MockProxyImpl(target, self))

    def register(self, expectation: Expectation) -> None:
        self._expectations[expectation.method_name].append(expectation)

    def find_expectation(self, method_name: str) -> Expectation | None:
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
                unsatisfied.append(
                    f"missing {len(expectations)} call(s) to '{method_name}'"
                )

        if unsatisfied:
            msg = f"Unsatisfied expectations:\n{'\n'.join(unsatisfied)}"
            raise AssertionError(msg)

    def reset(self) -> None:
        self._expectations.clear()


@final
class _MockProxyImpl[T]:
    def __init__(self, target: type[T], handler: ExpectationFinder) -> None:
        self._target = target
        self._handler = handler

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") and not name.startswith("__"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        def _mock_method(*args: Any, **kwargs: Any) -> Any:
            expectation = self._handler.find_expectation(name)

            if expectation is None:
                msg = (
                    f"Unexpected call to '{name}' with args={args}, kwargs={kwargs}. "
                    f"No expectation was set for this call."
                )
                raise AssertionError(msg)

            self._assert_expectation_args(expectation, name, args, kwargs)

            if expectation.kind == ExpectionKind.FUNCTION:
                if expectation.exception is not None:
                    raise expectation.exception
                return expectation.return_value

            if expectation.kind == ExpectionKind.COROUTINE:
                if expectation.exception is not None:
                    failed_future: asyncio.Future[Never] = asyncio.Future()
                    failed_future.set_exception(expectation.exception)
                    return failed_future

                succeeded_future: asyncio.Future[Any] = asyncio.Future()
                succeeded_future.set_result(expectation.return_value)
                return succeeded_future

            if (
                expectation.kind == ExpectionKind.CONTEXT_MANAGER
                or expectation.kind == ExpectionKind.ASYNC_CONTEXT_MANAGER
            ):
                if expectation.exception is not None:
                    raise expectation.exception

                return contextlib.nullcontext(enter_result=expectation.return_value)

            # fallback to function
            return expectation.return_value

        return _mock_method

    def _assert_expectation_args(
        self,
        expectation: Expectation,
        method_name: str,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
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
                diffs.append(
                    f"expected {key}={want_value!r}, but got {key}={got_value!r}"
                )

        for key in got_kwargs.keys():
            if key not in want_kwargs:
                diffs.append(f"unexpected {key}={got_kwargs[key]!r}")

        if diffs:
            raise AssertionError(
                f"Unexpected args for '{method_name}':\n{'\n'.join(diffs)}"
            )

    def _convert_args_to_kwargs(
        self, method_name: str, args: tuple[Any, ...]
    ) -> dict[str, Any]:
        target_method = getattr(self._target, method_name, None)
        if not target_method:
            raise AssertionError(
                f"Method '{method_name}' not found on target type {self._target}"
            )

        if not args:
            return {}

        signature = inspect.signature(target_method)
        param_names = [p for p in signature.parameters if p != "self"]

        return dict(zip(param_names, args))


@final
class Mockamorph[TargetT]:
    def __init__(self, target: type[TargetT]) -> None:
        self._target = target
        self._ctrl = MockController(target)

    def get_mock(self) -> TargetT:
        return self._ctrl.mock

    def expect[**MethodParam, ReturnT](
        self, method: Callable[Concatenate[TargetT, MethodParam], ReturnT]
    ) -> CallArgsSetter[MethodParam, ReturnT]:
        if method.__name__.startswith("_"):
            raise AttributeError("Cannot set expectations on private attribute")

        return CallArgsSetter[MethodParam, ReturnT](
            Expectation(method_name=method.__name__),
            self._ctrl,
        )

    def verify(self) -> None:
        self._ctrl.verify()

    def reset(self) -> None:
        self._ctrl.reset()

    def __enter__(self) -> Mockamorph[TargetT]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        _ = exc_type, exc_val, exc_tb
        self.verify()

    async def __aenter__(self) -> Mockamorph[TargetT]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        _ = exc_type, exc_val, exc_tb
        self.verify()
