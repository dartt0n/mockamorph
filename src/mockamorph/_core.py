from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Never,
    Protocol,
    cast,
    final,
)


@dataclass(frozen=False, kw_only=True, slots=True)
class Expectation:
    method_name: str
    awaitable: bool = False
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
class ReturnSetter:
    def __init__(self, expectation: Expectation, registrar: Registrar) -> None:
        self._expectation = expectation
        self._registrar = registrar

    def returns(self, *values: Any) -> None:
        self._expectation.return_value = values[0] if len(values) == 1 else values
        self._expectation.exception = None

        self._registrar.register(self._expectation)

    def raises(self, exception: BaseException) -> None:
        self._expectation.return_value = None
        self._expectation.exception = exception

        self._registrar.register(self._expectation)


@final
class CallArgsSetter:
    def __init__(self, expectation: Expectation, registrar: Registrar) -> None:
        self._expectation = expectation
        self._registrar = registrar

    def called_with(self, *args: Any, **kwargs: Any) -> ReturnSetter:
        self._expectation.args = args
        self._expectation.kwargs = kwargs
        self._expectation.awaitable = False
        return ReturnSetter(self._expectation, self._registrar)

    def awaited_with(self, *args: Any, **kwargs: Any) -> ReturnSetter:
        self._expectation.args = args
        self._expectation.kwargs = kwargs
        self._expectation.awaitable = True
        return ReturnSetter(self._expectation, self._registrar)


@final
class MethodProxy:
    def __init__(self, method_name: str, registrar: Registrar) -> None:
        self._method_name = method_name
        self._registrar = registrar

    def __call__(self) -> CallArgsSetter:
        return CallArgsSetter(
            Expectation(method_name=self._method_name), self._registrar
        )


@final
class ExpectationBuilder:
    # todo: infer typing and function annotation for static checking

    def __init__(self, registrar: Registrar) -> None:
        self._registrar = registrar

    def __getattr__(self, name: str) -> MethodProxy:
        if name.startswith("_"):
            raise AttributeError(
                f"Cannot set expectations on private attribute: {name}"
            )
        return MethodProxy(name, self._registrar)


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
                    f"{method_name}: {len(expectations)} unsatisfied expectation(s)"
                )

        if unsatisfied:
            msg = "Unsatisfied expectations:\n" + "\n".join(
                f"  - {e}" for e in unsatisfied
            )
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

            if expectation.exception is not None:
                if not expectation.awaitable:
                    raise expectation.exception

                failed_future: asyncio.Future[Never] = asyncio.Future()
                failed_future.set_exception(expectation.exception)
                return failed_future

            if expectation.awaitable:
                succeeded_future: asyncio.Future[Any] = asyncio.Future()
                succeeded_future.set_result(expectation.return_value)
                return succeeded_future

            return expectation.return_value

        return _mock_method

    def _assert_expectation_args(
        self,
        expectation: Expectation,
        method_name: str,
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
    ):
        want_kwargs = self._convert_args_to_kwargs(method_name, expectation.args)
        want_kwargs.update(expectation.kwargs)

        got_kwargs = self._convert_args_to_kwargs(method_name, call_args)
        got_kwargs.update(call_kwargs)

        diffs: list[str] = []

        for key, want_value in want_kwargs.items():
            if key not in got_kwargs:
                diffs.append(f"expected {key}={want_value}, but '{key}' is missing")
                continue

            got_value = got_kwargs[key]
            if got_value != want_value:
                diffs.append(f"expected {key}={want_value}, but got {key}={got_value}")

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
class Mockamorph[T]:
    def __init__(self, target: type[T]) -> None:
        self._ctrl = MockController(target)

    def get_mock(self) -> T:
        return self._ctrl.mock

    def expect(self) -> ExpectationBuilder:
        return ExpectationBuilder(self._ctrl)

    def verify(self) -> None:
        self._ctrl.verify()

    def reset(self) -> None:
        self._ctrl.reset()

    def __enter__(self) -> Mockamorph[T]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        _ = exc_type, exc_val, exc_tb
        self.verify()

    async def __aenter__(self) -> Mockamorph[T]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        _ = exc_type, exc_val, exc_tb
        self.verify()
