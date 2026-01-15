# Mockamorph

Lightweight interface mocking library for Python for expectation-driven testing.

> [!NOTE]
> I hate monkey-patching using string-literals in tests. I created `mockamorph` library to simply testing code, inspired by [uber-go/mock](https://github.com/uber-go/mock).

> [!WARNING]
> Code is written with AI assistance. List of tools used:
> - Zed Editor with Claude Opus 4.5.


## Quick Example

```python
from typing import Protocol
from mockamorph import Mockamorph


# 1. Define an interface (Protocol or ABC)
class UserRepository(Protocol):
    def get_user(self, user_id: int) -> str: ...
    def save_user(self, name: str) -> bool: ...


# 2. Your code that depends on the interface
class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    def greet_user(self, user_id: int) -> str:
        name = self.repo.get_user(user_id)
        return f"Hello, {name}!"

    def create_user(self, name: str) -> str:
        if self.repo.save_user(name):
            return "User created"
        raise RuntimeError("Failed to save user")


# 3. Test with Mockamorph
def test_user_service():
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
```

## Motivation

When we write code with SOLID principles in mind, there are many interfaces and usecases in our code that depend on interfaces. In production, we use adapters as concrete implementations for those interfaces, but in tests we need to rely on mocks in order to test business logic of usecases.

Typical code looks like this:
```python
class UserRepository(Protocol):
    def get_user(self, user_id: UserID) -> User | None: ...
    def save_user(self, user: User) -> User: ...
    
@final
class CreateNewUserUsecase:
    def __init__(self, repo: UserRepository):
        self.repo = repo
        
    def create_user(self, email: str) -> User:
        ... # some business logic
        
        user = User(email=email, token=10, ...) 
        user = self.repo.save_user(user)
        
        ... # some business logic
        
        return user
```


In order to test such code, we need to write the following code:
```python
class TestCreateNewUserUsecase(unittest.TestCase):
    def test_create_user(self):
        mock_repo = Mock()
        
        usecase = CreateNewUserUsecase(repo=mock_repo)
        email = "test@example.com"
        expected = User(email=email, token=10, id=1)

        mock_repo.save_user.return_value = expected
        result = usecase.create_user(email=email)
        self.assertEqual(result, expected)

        mock_repo.save_user.assert_called_once_with(User(email=email, token=10))
```

Note, that we need to:
- Setup mock before initalizing the `CreateNewUserUsecase` class
- Setup mocked return value right before actual call
- Assert that the method was called with the correct arguments after the execution
- Assert that the return value is correct

This way, we need to interact with `mock` object multiple times, increasing the complexity of the test and possibly of human error. To simplify this process, the `Mockamorph` library was created.

The same test could be written using `Mockamorph`:
```python
def test_create_user():
    email = "test@example.com"
    
    with Mockamorph(UserRepository) as ctrl:
        ctrl.expect().save_user().called_with(
            User(email=email, token=10)
        ).returns(
            User(email=email, token=10, id=1)
        )
        
        usecase = CreateNewUserUsecase(ctrl.get_mock())
        usecase.create_user(email)
        # Mockamorph automatically verifies all expectations were satisfied
```

Additionally, this approach simplifies TDT (table driven tests) by allowing to create mocks before actual test execution.


Some toy example:
```python
from collections.abc import Callable
from typing import Protocol, TypedDict, final

from mockamorph import Mockamorph


class Greeter(Protocol):
    def greet(self, name: str) -> str: ...


@final
class GreetUsecase:
    def __init__(self, greeter: Greeter) -> None:
        self._greeter = greeter

    def execute(self, name: str | None) -> str:
        if name is None:
            return "Hello, anon!"

        return self._greeter.greet(name) + "!"


def test_greet_table_driven() -> None:
    class Test(TypedDict):
        name: str
        mock: Callable[[Mockamorph[Greeter]], None]
        input: str | None
        expected: str

    tests: list[Test] = [
        {
            "name": "greets alice",
            "mock": lambda m: m.expect().greet().called_with("Alice").returns("Hello, Alice"),
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
            "mock": lambda m: m.expect().greet().called_with("").returns("Hello, stranger"),
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

```

## Examples

### Basic Mocking

```python
from mockamorph import Mockamorph

class Calculator(Protocol):
    def add(self, a: int, b: int) -> int: ...

with Mockamorph(Calculator) as mock:
    mock.expect().add().called_with(2, 3).returns(5)
    
    calc = mock.get_mock()
    assert calc.add(2, 3) == 5
```

### Multiple Return Values (FIFO)

```python
with Mockamorph(Calculator) as mock:
    mock.expect().add().called_with(1, 1).returns(2)
    mock.expect().add().called_with(1, 1).returns(3)  # Different return for same args
    
    calc = mock.get_mock()
    assert calc.add(1, 1) == 2  # First call
    assert calc.add(1, 1) == 3  # Second call
```

### Raising Exceptions

```python
class FileReader(Protocol):
    def read(self, path: str) -> str: ...

with Mockamorph(FileReader) as mock:
    mock.expect().read().called_with("/missing").raises(FileNotFoundError("Not found"))
    
    reader = mock.get_mock()
    with pytest.raises(FileNotFoundError):
        reader.read("/missing")
```

### Returning Tuples

```python
class DataSource(Protocol):
    def fetch(self) -> tuple[int, str, bool]: ...

with Mockamorph(DataSource) as mock:
    # Use multiple arguments to returns() for tuple unpacking
    mock.expect().fetch().called_with().returns(42, "hello", True)
    
    source = mock.get_mock()
    x, y, z = source.fetch()
    assert (x, y, z) == (42, "hello", True)
```

### Manual Verification

```python
mock = Mockamorph(Calculator)
mock.expect().add().called_with(1, 2).returns(3)

calc = mock.get_mock()
calc.add(1, 2)

mock.verify()  # Manually verify all expectations were satisfied
```

### Resetting Expectations

```python
mock = Mockamorph(Calculator)
mock.expect().add().called_with(1, 2).returns(3)
mock.reset()  # Clear all expectations
mock.verify()  # Passes - no expectations to satisfy
```

### Async Support

```python
class RemoveServer(Protocol):
    async def fetch(self, resource: str) -> bytes: ...

async with Mockamorph(RemoveServer) as mock:
    mock.expect().fetch().awaited_with(resource="resA").returns(b"ok")
    
    source = mock.get_mock()
    assert await source.fetch() == b"ok"
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/mockamorph/mockamorph.git
cd mockamorph

# Install dependencies with uv
uv sync --all-groups
```

### Running Tests

```bash
uv run pytest .
```

### Type Checking

```bash
uv run mypy src
# or
uv run basedpyright
```

### Building

```bash
uv run hatch build
```

### Publishing

```bash
uv run hatch publish
```

## License

MIT License - see LICENSE file for details.
