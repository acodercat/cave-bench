"""
Object Types - Stateful Benchmark

Tests: Custom classes with attributes and methods.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime, Type
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# CUSTOM CLASSES
# ============================================================================

class Person:
    def __init__(self, name: str = "", age: int = 0, city: str = ""):
        self.name = name
        self.age = age
        self.city = city

    def introduce(self) -> str:
        return f"Hi, I'm {self.name}, {self.age} years old from {self.city}."

    def have_birthday(self):
        self.age += 1


class BankAccount:
    def __init__(self, balance: float = 0.0):
        self.balance = balance
        self.transactions = []

    def deposit(self, amount: float):
        self.balance += amount
        self.transactions.append(f"+{amount}")

    def withdraw(self, amount: float):
        self.balance -= amount
        self.transactions.append(f"-{amount}")

    def apply_interest(self, rate: float):
        interest = self.balance * rate
        self.balance += interest


class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, item: dict):
        self.items.append(item)

    def remove_item(self, name: str):
        self.items = [i for i in self.items if i.get("name") != name]

    def get_total(self) -> float:
        return sum(item.get("price", 0) * item.get("quantity", 1) for item in self.items)

    def clear(self):
        self.items = []


class Counter:
    def __init__(self, value: int = 0):
        self.value = value

    def increment(self):
        self.value += 1

    def decrement(self):
        self.value -= 1

    def add(self, n: int):
        self.value += n

    def reset(self):
        self.value = 0


class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if self._items:
            return self._items.pop()
        return None

    def peek(self):
        if self._items:
            return self._items[-1]
        return None

    def size(self) -> int:
        return len(self._items)


# ============================================================================
# INITIAL STATE
# ============================================================================

person = Person()
account = BankAccount(balance=1000.0)
cart = ShoppingCart()
counter = Counter(value=0)
stack = Stack()

# Result variables (reused)
result_str = ""
result_num = 0.0


# ============================================================================
# VALIDATORS - Person
# ============================================================================

def validate_person_init(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    p = runtime.get_variable("person")
    ok = p.name == "Alice" and p.age == 25
    return ValidatorResult(ok, f"name='{p.name}', age={p.age}")

def validate_person_update(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    p = runtime.get_variable("person")
    ok = p.age == 26 and p.city == "Boston"
    return ValidatorResult(ok, f"age={p.age}, city='{p.city}'")

def validate_person_method(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_str")
    expected = "Hi, I'm Alice, 26 years old from Boston."
    return ValidatorResult(r == expected, f"result_str = '{r}'")


def validate_person_birthday_init(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    p = runtime.get_variable("person")
    ok = p.name == "Bob" and p.age == 30
    return ValidatorResult(ok, f"name='{p.name}', age={p.age}")

def validate_person_birthday(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    p = runtime.get_variable("person")
    return ValidatorResult(p.age == 31, f"age = {p.age}, expected 31")

def validate_person_birthday2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    p = runtime.get_variable("person")
    return ValidatorResult(p.age == 32, f"age = {p.age}, expected 32")


# ============================================================================
# VALIDATORS - BankAccount
# ============================================================================

def validate_account_deposit(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    a = runtime.get_variable("account")
    ok = abs(a.balance - 1500.0) < 0.01
    return ValidatorResult(ok, f"balance = {a.balance}")

def validate_account_withdraw(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    a = runtime.get_variable("account")
    ok = abs(a.balance - 1300.0) < 0.01
    return ValidatorResult(ok, f"balance = {a.balance}")

def validate_account_interest(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    a = runtime.get_variable("account")
    expected = 1300.0 * 1.10
    ok = abs(a.balance - expected) < 0.01
    return ValidatorResult(ok, f"balance = {a.balance}")


def validate_account_multi_deposit(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    a = runtime.get_variable("account")
    ok = abs(a.balance - 1300.0) < 0.01
    return ValidatorResult(ok, f"balance = {a.balance}")

def validate_account_multi_withdraw(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    a = runtime.get_variable("account")
    ok = abs(a.balance - 1050.0) < 0.01
    return ValidatorResult(ok, f"balance = {a.balance}")

def validate_account_balance(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_num")
    ok = abs(r - 1050.0) < 0.01
    return ValidatorResult(ok, f"result_num = {r}")


# ============================================================================
# VALIDATORS - ShoppingCart
# ============================================================================

def validate_cart_add(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    c = runtime.get_variable("cart")
    ok = len(c.items) == 1 and c.items[0].get("name") == "Book"
    return ValidatorResult(ok, f"items = {c.items}")

def validate_cart_add2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    c = runtime.get_variable("cart")
    ok = len(c.items) == 2
    return ValidatorResult(ok, f"items count = {len(c.items)}")

def validate_cart_total(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_num")
    expected = 15.99 + 2.50
    ok = abs(r - expected) < 0.01
    return ValidatorResult(ok, f"result_num = {r}")


def validate_cart_qty_add(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    c = runtime.get_variable("cart")
    ok = len(c.items) == 1 and c.items[0].get("quantity") == 3
    return ValidatorResult(ok, f"items = {c.items}")

def validate_cart_qty_add2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    c = runtime.get_variable("cart")
    ok = len(c.items) == 2
    return ValidatorResult(ok, f"items count = {len(c.items)}")

def validate_cart_qty_total(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_num")
    expected = 10.0 * 3 + 5.0 * 2
    ok = abs(r - expected) < 0.01
    return ValidatorResult(ok, f"result_num = {r}")


# ============================================================================
# VALIDATORS - Counter
# ============================================================================

def validate_counter_inc(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    c = runtime.get_variable("counter")
    return ValidatorResult(c.value == 3, f"value = {c.value}")

def validate_counter_mixed(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    c = runtime.get_variable("counter")
    return ValidatorResult(c.value == 4, f"value = {c.value}")

def validate_counter_reset(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    c = runtime.get_variable("counter")
    return ValidatorResult(c.value == 0, f"value = {c.value}")


def validate_counter_add10(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    c = runtime.get_variable("counter")
    return ValidatorResult(c.value == 10, f"value = {c.value}")

def validate_counter_add5(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    c = runtime.get_variable("counter")
    return ValidatorResult(c.value == 15, f"value = {c.value}")

def validate_counter_store(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_num")
    return ValidatorResult(r == 15, f"result_num = {r}")


# ============================================================================
# VALIDATORS - Stack
# ============================================================================

def validate_stack_push(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    s = runtime.get_variable("stack")
    ok = s.size() == 3
    return ValidatorResult(ok, f"size = {s.size()}")

def validate_stack_pop(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    s = runtime.get_variable("stack")
    r = runtime.get_variable("result_str")
    ok = s.size() == 2 and r == "C"
    return ValidatorResult(ok, f"size = {s.size()}, popped = {r}")

def validate_stack_peek(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    s = runtime.get_variable("stack")
    r = runtime.get_variable("result_str")
    ok = s.size() == 2 and r == "B"
    return ValidatorResult(ok, f"size = {s.size()}, top = {r}")


def validate_stack_push_nums(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    s = runtime.get_variable("stack")
    ok = s.size() == 5
    return ValidatorResult(ok, f"size = {s.size()}")

def validate_stack_pop_sum(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    s = runtime.get_variable("stack")
    r = runtime.get_variable("result_num")
    # Popped 5 + 4 = 9
    ok = s.size() == 3 and r == 9
    return ValidatorResult(ok, f"size = {s.size()}, sum = {r}")

def validate_stack_size(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    r = runtime.get_variable("result_num")
    return ValidatorResult(r == 3, f"result_num = {r}")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("person", person, "Person object. Attributes: name (str), age (int), city (str). Methods: introduce() -> str, have_birthday()."),
    Variable("account", account, "BankAccount with balance=1000.0. Methods: deposit(amount), withdraw(amount), apply_interest(rate)."),
    Variable("cart", cart, "ShoppingCart with empty items list. Methods: add_item(dict), remove_item(name), get_total() -> float, clear()."),
    Variable("counter", counter, "Counter with value=0. Methods: increment(), decrement(), add(n), reset()."),
    Variable("stack", stack, "Stack (LIFO). Methods: push(item), pop() -> item, peek() -> item, size() -> int."),
    Variable("result_str", result_str, "Store string results here."),
    Variable("result_num", result_num, "Store numeric results here."),
]

types = [
    Type(Person),
    Type(BankAccount),
    Type(ShoppingCart),
    Type(Counter),
    Type(Stack),
]

validators = {
    # Person
    "validate_person_init": validate_person_init,
    "validate_person_update": validate_person_update,
    "validate_person_method": validate_person_method,
    "validate_person_birthday_init": validate_person_birthday_init,
    "validate_person_birthday": validate_person_birthday,
    "validate_person_birthday2": validate_person_birthday2,
    # BankAccount
    "validate_account_deposit": validate_account_deposit,
    "validate_account_withdraw": validate_account_withdraw,
    "validate_account_interest": validate_account_interest,
    "validate_account_multi_deposit": validate_account_multi_deposit,
    "validate_account_multi_withdraw": validate_account_multi_withdraw,
    "validate_account_balance": validate_account_balance,
    # ShoppingCart
    "validate_cart_add": validate_cart_add,
    "validate_cart_add2": validate_cart_add2,
    "validate_cart_total": validate_cart_total,
    "validate_cart_qty_add": validate_cart_qty_add,
    "validate_cart_qty_add2": validate_cart_qty_add2,
    "validate_cart_qty_total": validate_cart_qty_total,
    # Counter
    "validate_counter_inc": validate_counter_inc,
    "validate_counter_mixed": validate_counter_mixed,
    "validate_counter_reset": validate_counter_reset,
    "validate_counter_add10": validate_counter_add10,
    "validate_counter_add5": validate_counter_add5,
    "validate_counter_store": validate_counter_store,
    # Stack
    "validate_stack_push": validate_stack_push,
    "validate_stack_pop": validate_stack_pop,
    "validate_stack_peek": validate_stack_peek,
    "validate_stack_push_nums": validate_stack_push_nums,
    "validate_stack_pop_sum": validate_stack_pop_sum,
    "validate_stack_size": validate_stack_size,
}
