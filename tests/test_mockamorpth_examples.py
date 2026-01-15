from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypedDict, final

import pytest

from mockamorph import Mockamorph


class UserRepository(Protocol):
    """User repository interface for the quick example."""

    def get_user(self, user_id: int) -> str: ...
    def save_user(self, name: str) -> bool: ...


class UserService:
    """User service that depends on UserRepository."""

    def __init__(self, repo: UserRepository):
        self.repo = repo

    def greet_user(self, user_id: int) -> str:
        name = self.repo.get_user(user_id)
        return f"Hello, {name}!"

    def create_user(self, name: str) -> str:
        if self.repo.save_user(name):
            return "User created"
        raise RuntimeError("Failed to save user")


def test_quick_example_user_service() -> None:
    """Test the quick example from README."""
    with Mockamorph(UserRepository) as mock:
        # Set expectations BEFORE calling code
        mock.expect().get_user().called_with(42).returns("Alice")
        mock.expect().save_user().called_with("Bob").returns(True)
        mock.expect().save_user().called_with("").raises(RuntimeError("Invalid name"))

        # Use the mock
        service = UserService(mock.get_mock())

        assert service.greet_user(42) == "Hello, Alice!"
        assert service.create_user("Bob") == "User created"

        with pytest.raises(RuntimeError, match="Invalid name"):
            service.create_user("")
        # Mockamorph auto-verifies all expectations were satisfied on exit


@dataclass
class UserID:
    """User ID value object."""

    value: int


@dataclass
class User:
    """User entity."""

    email: str
    token: int = 10
    id: int | None = None


class UserRepositoryForMotivation(Protocol):
    """User repository for motivation example."""

    def get_user(self, user_id: UserID) -> User | None: ...
    def save_user(self, user: User) -> User: ...


@final
class CreateNewUserUsecase:
    """Use case for creating new users."""

    def __init__(self, repo: UserRepositoryForMotivation):
        self.repo = repo

    def create_user(self, email: str) -> User:
        # some business logic
        user = User(email=email, token=10)
        user = self.repo.save_user(user)
        # some business logic
        return user


def test_motivation_example_create_user() -> None:
    """Test the motivation example from README."""
    email = "test@example.com"

    with Mockamorph(UserRepositoryForMotivation) as ctrl:
        ctrl.expect().save_user().called_with(User(email=email, token=10)).returns(
            User(email=email, token=10, id=1)
        )

        usecase = CreateNewUserUsecase(ctrl.get_mock())
        result = usecase.create_user(email)

        assert result == User(email=email, token=10, id=1)


class Greeter(Protocol):
    """Greeter interface for table driven tests."""

    def greet(self, name: str) -> str: ...


@final
class GreetUsecase:
    """Use case for greeting users."""

    def __init__(self, greeter: Greeter) -> None:
        self._greeter = greeter

    def execute(self, name: str | None) -> str:
        if name is None:
            return "Hello, anon!"

        return self._greeter.greet(name) + "!"


def test_greet_table_driven() -> None:
    """Test the table driven tests example from README."""

    class Test(TypedDict):
        name: str
        mock: Callable[[Mockamorph[Greeter]], None]
        input: str | None
        expected: str

    tests: list[Test] = [
        {
            "name": "greets alice",
            "mock": lambda m: m.expect()
            .greet()
            .called_with("Alice")
            .returns("Hello, Alice"),
            "input": "Alice",
            "expected": "Hello, Alice!",
        },
        {
            "name": "greets bob",
            "mock": lambda m: m.expect().greet().called_with("Bob").returns("Hi, Bob"),
            "input": "Bob",
            "expected": "Hi, Bob!",
        },
        {
            "name": "greets empty",
            "mock": lambda m: m.expect()
            .greet()
            .called_with("")
            .returns("Hello, stranger"),
            "input": "",
            "expected": "Hello, stranger!",
        },
        {
            "name": "name is missing",
            "mock": lambda m: None,  # no calls expected
            "input": None,
            "expected": "Hello, anon!",
        },
    ]

    for tt in tests:
        with Mockamorph(Greeter) as ctrl:
            tt["mock"](ctrl)
            result = GreetUsecase(ctrl.get_mock()).execute(tt["input"])
            assert result == tt["expected"], f"Failed: {tt['name']}"


class Calculator(Protocol):
    """Calculator interface for basic mocking example."""

    def add(self, a: int, b: int) -> int: ...


def test_basic_mocking() -> None:
    """Test the basic mocking example from README."""
    with Mockamorph(Calculator) as mock:
        mock.expect().add().called_with(2, 3).returns(5)

        calc = mock.get_mock()
        assert calc.add(2, 3) == 5


def test_multiple_return_values_fifo() -> None:
    """Test the multiple return values FIFO example from README."""
    with Mockamorph(Calculator) as mock:
        mock.expect().add().called_with(1, 1).returns(2)
        # Different return for same args
        mock.expect().add().called_with(1, 1).returns(3)

        calc = mock.get_mock()
        assert calc.add(1, 1) == 2  # First call
        assert calc.add(1, 1) == 3  # Second call


class FileReader(Protocol):
    """File reader interface for exception example."""

    def read(self, path: str) -> str: ...


def test_raising_exceptions() -> None:
    """Test the raising exceptions example from README."""
    with Mockamorph(FileReader) as mock:
        mock.expect().read().called_with("/missing").raises(
            FileNotFoundError("Not found")
        )

        reader = mock.get_mock()
        with pytest.raises(FileNotFoundError):
            reader.read("/missing")


class DataSource(Protocol):
    """Data source interface for tuple example."""

    def fetch(self) -> tuple[int, str, bool]: ...


def test_returning_tuples() -> None:
    """Test the returning tuples example from README."""
    with Mockamorph(DataSource) as mock:
        # Use multiple arguments to returns() for tuple unpacking
        mock.expect().fetch().called_with().returns(42, "hello", True)

        source = mock.get_mock()
        x, y, z = source.fetch()
        assert (x, y, z) == (42, "hello", True)


def test_manual_verification() -> None:
    """Test the manual verification example from README."""
    mock = Mockamorph(Calculator)
    mock.expect().add().called_with(1, 2).returns(3)

    calc = mock.get_mock()
    calc.add(1, 2)

    mock.verify()  # Manually verify all expectations were satisfied


def test_resetting_expectations() -> None:
    """Test the resetting expectations example from README."""
    mock = Mockamorph(Calculator)
    mock.expect().add().called_with(1, 2).returns(3)
    mock.reset()  # Clear all expectations
    mock.verify()  # Passes - no expectations to satisfy


class RemoteServer(Protocol):
    """Remote server interface for async example."""

    async def fetch(self, resource: str) -> bytes: ...


@pytest.mark.asyncio
async def test_async_support() -> None:
    """Test the async support example from README."""
    async with Mockamorph(RemoteServer) as mock:
        mock.expect().fetch().awaited_with(resource="resA").returns(b"ok")

        source = mock.get_mock()
        assert await source.fetch(resource="resA") == b"ok"
