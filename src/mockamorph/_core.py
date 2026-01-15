from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
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
        self._mock = cast(T, _MockProxyImpl(self))

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
class _MockProxyImpl:
    def __init__(self, handler: ExpectationFinder) -> None:
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

            if expectation.exception is not None:
                if not expectation.awaitable:
                    raise expectation.exception

                future = asyncio.Future()
                future.set_exception(expectation.exception)
                return future

            if not expectation.awaitable:
                return expectation.return_value

            future = asyncio.Future()
            future.set_result(expectation.return_value)
            return future

        return _mock_method


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
