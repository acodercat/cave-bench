"""
Multi-Variable (15) - Stateful Benchmark

Tests: Managing 15 variables of diverse types modified in each turn.
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# INITIAL STATE - 15 variables with diverse types
# ============================================================================

# Strings (3)
title = ""
author = ""
publisher = ""

# Integers (3)
year = 0
pages = 0
edition = 0

# Floats (3)
price = 0.0
rating = 0.0
weight = 0.0

# Booleans (3)
available = False
bestseller = False
ebook = False

# Collections (3)
genres = []
chapters = []
info = {}


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_book_init(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "title": "Python Guide",
        "author": "John Smith",
        "publisher": "TechBooks",
        "year": 2023,
        "pages": 450,
        "edition": 1,
        "price": 49.99,
        "rating": 4.5,
        "weight": 1.2,
        "available": True,
        "bestseller": False,
        "ebook": True,
        "genres": ["programming", "education"],
        "chapters": ["Intro", "Basics", "Advanced"],
        "info": {"isbn": "123-456", "language": "English"}
    }
    errors = []
    for key, exp_val in expected.items():
        val = runtime.get_variable(key)
        if isinstance(exp_val, float):
            if abs(val - exp_val) > 0.01:
                errors.append(f"{key}={val}")
        elif val != exp_val:
            errors.append(f"{key}={val}")
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Book initialized")

def validate_book_update(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "title": "Python Guide 2nd Edition",
        "author": "John Smith",
        "publisher": "TechBooks Pro",
        "year": 2024,
        "pages": 550,
        "edition": 2,
        "price": 59.99,
        "rating": 4.7,
        "weight": 1.4,
        "available": True,
        "bestseller": True,
        "ebook": True,
        "genres": ["programming", "education", "reference"],
        "chapters": ["Intro", "Basics", "Advanced", "Expert"],
        "info": {"isbn": "123-456", "language": "English", "updated": True}
    }
    errors = []
    for key, exp_val in expected.items():
        val = runtime.get_variable(key)
        if isinstance(exp_val, float):
            if abs(val - exp_val) > 0.01:
                errors.append(f"{key}={val}")
        elif val != exp_val:
            errors.append(f"{key}={val}")
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Book updated")

def validate_book_discount(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "title": "[SALE] Python Guide 2nd Edition",
        "author": "John Smith",
        "publisher": "TechBooks Pro",
        "year": 2024,
        "pages": 550,
        "edition": 2,
        "price": 47.99,
        "rating": 4.7,
        "weight": 1.4,
        "available": True,
        "bestseller": True,
        "ebook": True,
        "genres": ["programming", "education", "reference", "sale"],
        "chapters": ["Intro", "Basics", "Advanced", "Expert"],
        "info": {"isbn": "123-456", "language": "English", "updated": True, "discount": 0.2}
    }
    errors = []
    for key, exp_val in expected.items():
        val = runtime.get_variable(key)
        if isinstance(exp_val, float):
            if abs(val - exp_val) > 0.01:
                errors.append(f"{key}={val}")
        elif val != exp_val:
            errors.append(f"{key}={val}")
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Discount applied")


def validate_course_init(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "title": "Web Development",
        "author": "Jane Doe",
        "publisher": "LearnOnline",
        "year": 2024,
        "pages": 0,
        "edition": 1,
        "price": 199.0,
        "rating": 0.0,
        "weight": 0.0,
        "available": True,
        "bestseller": False,
        "ebook": False,
        "genres": ["web", "frontend", "backend"],
        "chapters": [],
        "info": {"hours": 40, "level": "beginner"}
    }
    errors = []
    for key, exp_val in expected.items():
        val = runtime.get_variable(key)
        if isinstance(exp_val, float):
            if abs(val - exp_val) > 0.01:
                errors.append(f"{key}={val}")
        elif val != exp_val:
            errors.append(f"{key}={val}")
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Course initialized")

def validate_course_progress(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "title": "Web Development",
        "author": "Jane Doe",
        "publisher": "LearnOnline",
        "year": 2024,
        "pages": 10,
        "edition": 1,
        "price": 199.0,
        "rating": 4.8,
        "weight": 0.0,
        "available": True,
        "bestseller": True,
        "ebook": False,
        "genres": ["web", "frontend", "backend"],
        "chapters": ["HTML", "CSS", "JavaScript"],
        "info": {"hours": 40, "level": "beginner", "students": 1000}
    }
    errors = []
    for key, exp_val in expected.items():
        val = runtime.get_variable(key)
        if isinstance(exp_val, float):
            if abs(val - exp_val) > 0.01:
                errors.append(f"{key}={val}")
        elif val != exp_val:
            errors.append(f"{key}={val}")
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Progress updated")

def validate_course_complete(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    expected = {
        "title": "Web Development [COMPLETE]",
        "author": "Jane Doe",
        "publisher": "LearnOnline",
        "year": 2024,
        "pages": 20,
        "edition": 2,
        "price": 249.0,
        "rating": 4.9,
        "weight": 0.0,
        "available": True,
        "bestseller": True,
        "ebook": True,
        "genres": ["web", "frontend", "backend", "fullstack"],
        "chapters": ["HTML", "CSS", "JavaScript", "React", "Node"],
        "info": {"hours": 40, "level": "beginner", "students": 1000, "complete": True}
    }
    errors = []
    for key, exp_val in expected.items():
        val = runtime.get_variable(key)
        if isinstance(exp_val, float):
            if abs(val - exp_val) > 0.01:
                errors.append(f"{key}={val}")
        elif val != exp_val:
            errors.append(f"{key}={val}")
    if errors:
        return ValidatorResult(False, "; ".join(errors[:3]))
    return ValidatorResult(True, "Course completed")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("title", title, "Item title (str, initial: '')."),
    Variable("author", author, "Author/creator name (str, initial: '')."),
    Variable("publisher", publisher, "Publisher name (str, initial: '')."),
    Variable("year", year, "Publication year (int, initial: 0)."),
    Variable("pages", pages, "Page count or lessons (int, initial: 0)."),
    Variable("edition", edition, "Edition number (int, initial: 0)."),
    Variable("price", price, "Price in dollars (float, initial: 0.0)."),
    Variable("rating", rating, "Average rating 0-5 (float, initial: 0.0)."),
    Variable("weight", weight, "Weight in kg (float, initial: 0.0)."),
    Variable("available", available, "Availability status (bool, initial: False)."),
    Variable("bestseller", bestseller, "Bestseller status (bool, initial: False)."),
    Variable("ebook", ebook, "Digital version available (bool, initial: False)."),
    Variable("genres", genres, "Category/genre tags (list, initial: [])."),
    Variable("chapters", chapters, "Chapter/section names (list, initial: [])."),
    Variable("info", info, "Additional metadata (dict, initial: {})."),
]

validators = {
    "validate_book_init": validate_book_init,
    "validate_book_update": validate_book_update,
    "validate_book_discount": validate_book_discount,
    "validate_course_init": validate_course_init,
    "validate_course_progress": validate_course_progress,
    "validate_course_complete": validate_course_complete,
}
