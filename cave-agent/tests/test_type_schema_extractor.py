"""Tests for TypeSchemaExtractor class."""

import pytest
from typing import List, Dict, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from cave_agent.python_runtime import TypeSchemaExtractor, Function, Variable
from pydantic import BaseModel, Field


# =============================================================================
# Test Fixtures - Enums
# =============================================================================

class Status(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


# =============================================================================
# Test Fixtures - Dataclasses
# =============================================================================

@dataclass
class SimpleDataclass:
    name: str
    age: int


@dataclass
class DataclassWithDefaults:
    name: str
    count: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class NestedDataclass:
    item: SimpleDataclass
    status: Status


# =============================================================================
# Test Fixtures - Pydantic Models
# =============================================================================

class SimplePydantic(BaseModel):
    name: str
    value: int


class PydanticWithDescription(BaseModel):
    title: str = Field(description="The title of the item")
    count: int = Field(default=0, description="Number of items")


class NestedPydantic(BaseModel):
    item: SimplePydantic
    priority: Priority


# =============================================================================
# Test Functions for Extraction
# =============================================================================

def func_with_builtin_types(a: int, b: str, c: float) -> bool:
    """Function with only builtin types."""
    return True


def func_with_enum(status: Status) -> Priority:
    """Function with enum types."""
    return Priority.HIGH


def func_with_dataclass(data: SimpleDataclass) -> NestedDataclass:
    """Function with dataclass types."""
    pass


def func_with_optional(name: Optional[str], count: Optional[int] = None) -> Optional[bool]:
    """Function with Optional types."""
    return None


def func_with_union(value: Union[str, int, float]) -> Union[bool, None]:
    """Function with Union types."""
    return True


def func_with_generic_collections(
    items: List[str],
    mapping: Dict[str, int],
    nested: List[Dict[str, List[int]]]
) -> Dict[str, Any]:
    """Function with generic collection types."""
    return {}


def func_with_callable(callback: Callable[[int, str], bool]) -> Callable:
    """Function with Callable type."""
    return lambda: None


def func_with_nested_dataclass(data: NestedDataclass) -> List[SimpleDataclass]:
    """Function with nested dataclass."""
    return []


def func_no_annotations(a, b):
    """Function without type annotations."""
    return a + b


def func_partial_annotations(a: int, b) -> str:
    """Function with partial annotations."""
    return str(a)


# =============================================================================
# Tests - Basic Functionality
# =============================================================================

class TestTypeSchemaExtractorBasics:
    """Test basic TypeSchemaExtractor functionality."""

    def test_extract_from_builtin_types_returns_empty(self):
        """Builtin types should not generate schemas."""
        schemas = TypeSchemaExtractor.extract_from_signature(func_with_builtin_types)
        assert schemas == {}

    def test_extract_from_no_annotations_returns_empty(self):
        """Functions without annotations should return empty schemas."""
        schemas = TypeSchemaExtractor.extract_from_signature(func_no_annotations)
        assert schemas == {}

    def test_extract_from_partial_annotations(self):
        """Functions with partial annotations should work."""
        schemas = TypeSchemaExtractor.extract_from_signature(func_partial_annotations)
        assert schemas == {}  # Only builtin types



# =============================================================================
# Tests - Enum Extraction
# =============================================================================

class TestEnumExtraction:
    """Test Enum schema extraction."""

    def test_extract_enum_from_parameter(self):
        """Enum in parameter should be extracted."""
        schemas = TypeSchemaExtractor.extract_from_signature(func_with_enum)
        assert "Status" in schemas
        assert "PENDING" in schemas["Status"]
        assert "ACTIVE" in schemas["Status"]
        assert "COMPLETED" in schemas["Status"]

    def test_extract_enum_from_return_type(self):
        """Enum in return type should be extracted."""
        schemas = TypeSchemaExtractor.extract_from_signature(func_with_enum)
        assert "Priority" in schemas
        assert "LOW" in schemas["Priority"]
        assert "MEDIUM" in schemas["Priority"]
        assert "HIGH" in schemas["Priority"]

    def test_enum_format(self):
        """Enum schema should have correct format."""
        schemas = TypeSchemaExtractor.extract_from_signature(func_with_enum)
        status_schema = schemas["Status"]

        assert "Status (Enum):" in status_schema
        assert "PENDING = 'pending'" in status_schema
        assert "ACTIVE = 'active'" in status_schema


# =============================================================================
# Tests - Dataclass Extraction
# =============================================================================

class TestDataclassExtraction:
    """Test dataclass schema extraction."""

    def test_extract_simple_dataclass(self):
        """Simple dataclass should be extracted."""
        schemas = TypeSchemaExtractor.extract_from_signature(func_with_dataclass)
        assert "SimpleDataclass" in schemas

        schema = schemas["SimpleDataclass"]
        assert "name: str" in schema
        assert "age: int" in schema

    def test_extract_dataclass_with_defaults(self):
        """Dataclass with defaults should show default values."""
        def func(data: DataclassWithDefaults) -> None:
            pass

        schemas = TypeSchemaExtractor.extract_from_signature(func)
        assert "DataclassWithDefaults" in schemas

        schema = schemas["DataclassWithDefaults"]
        assert "name: str" in schema
        assert "count: int = 0" in schema
        assert "tags: list[str] = <factory>" in schema

    def test_extract_nested_dataclass(self):
        """Nested dataclass should extract all types."""
        schemas = TypeSchemaExtractor.extract_from_signature(func_with_nested_dataclass)

        # Should extract the nested dataclass
        assert "NestedDataclass" in schemas
        assert "SimpleDataclass" in schemas
        assert "Status" in schemas

    def test_dataclass_format(self):
        """Dataclass schema should have correct format."""
        schemas = TypeSchemaExtractor.extract_from_signature(func_with_dataclass)
        schema = schemas["SimpleDataclass"]

        assert "SimpleDataclass:" in schema
        assert "  - name: str" in schema
        assert "  - age: int" in schema


# =============================================================================
# Tests - Generic Types
# =============================================================================

class TestGenericTypeHandling:
    """Test handling of generic types."""

    def test_optional_type_formatting(self):
        """Optional types should be formatted correctly."""
        type_str = TypeSchemaExtractor._format_type_annotation(Optional[str])
        assert type_str == "Optional[str]"

    def test_union_type_formatting(self):
        """Union types should be formatted correctly."""
        type_str = TypeSchemaExtractor._format_type_annotation(Union[str, int, float])
        assert "Union[" in type_str
        assert "str" in type_str
        assert "int" in type_str
        assert "float" in type_str

    def test_list_type_formatting(self):
        """List types should be formatted correctly."""
        type_str = TypeSchemaExtractor._format_type_annotation(List[str])
        assert type_str == "list[str]"

    def test_dict_type_formatting(self):
        """Dict types should be formatted correctly."""
        type_str = TypeSchemaExtractor._format_type_annotation(Dict[str, int])
        assert type_str == "dict[str, int]"

    def test_nested_generic_formatting(self):
        """Nested generics should be formatted correctly."""
        type_str = TypeSchemaExtractor._format_type_annotation(List[Dict[str, List[int]]])
        assert "list[dict[str, list[int]]]" == type_str

    def test_generic_with_custom_type_extracts_inner(self):
        """Generic containing custom type should extract the inner type."""
        def func(items: List[SimpleDataclass]) -> Dict[str, Status]:
            pass

        schemas = TypeSchemaExtractor.extract_from_signature(func)
        assert "SimpleDataclass" in schemas
        assert "Status" in schemas

    def test_optional_with_custom_type_extracts_inner(self):
        """Optional containing custom type should extract the inner type."""
        def func(item: Optional[SimpleDataclass]) -> Optional[Status]:
            pass

        schemas = TypeSchemaExtractor.extract_from_signature(func)
        assert "SimpleDataclass" in schemas
        assert "Status" in schemas


# =============================================================================
# Tests - Callable Handling
# =============================================================================

class TestCallableHandling:
    """Test handling of Callable types."""

    def test_callable_type_formatting(self):
        """Callable should be formatted as 'Callable'."""
        type_str = TypeSchemaExtractor._format_type_annotation(Callable)
        assert type_str == "Callable"

    def test_callable_does_not_crash(self):
        """Functions with Callable params should not crash."""
        schemas = TypeSchemaExtractor.extract_from_signature(func_with_callable)
        # Should complete without error, no schema for Callable itself
        assert isinstance(schemas, dict)


# =============================================================================
# Tests - Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_none_type_formatting(self):
        """None type should be formatted as 'None'."""
        type_str = TypeSchemaExtractor._format_type_annotation(None)
        assert type_str == "None"

        type_str = TypeSchemaExtractor._format_type_annotation(type(None))
        assert type_str == "None"

    def test_string_annotation_handling(self):
        """String annotations should be handled gracefully."""
        type_str = TypeSchemaExtractor._format_type_annotation("SomeForwardRef")
        assert type_str == "SomeForwardRef"

    def test_any_type_formatting(self):
        """Any type should be formatted correctly."""
        type_str = TypeSchemaExtractor._format_type_annotation(Any)
        # Any doesn't have __name__, falls back to str()
        assert "Any" in type_str

    def test_cycle_prevention(self):
        """Recursive types should not cause infinite loops."""
        @dataclass
        class Node:
            value: int
            children: List["Node"]  # Forward reference

        def func(node: Node) -> None:
            pass

        # Should complete without infinite recursion
        schemas = TypeSchemaExtractor.extract_from_signature(func)
        assert "Node" in schemas

    def test_empty_function(self):
        """Function with no params and no return should work."""
        def empty_func():
            pass

        schemas = TypeSchemaExtractor.extract_from_signature(empty_func)
        assert schemas == {}


# =============================================================================
# Tests - Pydantic Models
# =============================================================================

class TestPydanticExtraction:
    """Test Pydantic model schema extraction."""

    def test_extract_simple_pydantic(self):
        """Simple Pydantic model should be extracted."""
        def func(data: SimplePydantic) -> None:
            pass

        schemas = TypeSchemaExtractor.extract_from_signature(func)
        assert "SimplePydantic" in schemas

        schema = schemas["SimplePydantic"]
        assert "name: str" in schema
        assert "value: int" in schema

    def test_extract_pydantic_with_description(self):
        """Pydantic field descriptions should be included."""
        def func(data: PydanticWithDescription) -> None:
            pass

        schemas = TypeSchemaExtractor.extract_from_signature(func)
        assert "PydanticWithDescription" in schemas

        schema = schemas["PydanticWithDescription"]
        assert "title: str" in schema
        assert "The title of the item" in schema
        assert "count: int" in schema
        assert "Number of items" in schema

    def test_extract_nested_pydantic(self):
        """Nested Pydantic models should extract all types."""
        def func(data: NestedPydantic) -> None:
            pass

        schemas = TypeSchemaExtractor.extract_from_signature(func)
        assert "NestedPydantic" in schemas
        assert "SimplePydantic" in schemas
        assert "Priority" in schemas


# =============================================================================
# Tests - Function Class Integration
# =============================================================================

class TestFunctionIntegration:
    """Test TypeSchemaExtractor integration with Function class."""

    def test_function_extracts_type_schemas(self):
        """Function class should use TypeSchemaExtractor."""
        func = Function(func_with_enum)

        assert "Status" in func.type_schemas
        assert "Priority" in func.type_schemas

    def test_function_str_excludes_type_schemas(self):
        """Function.__str__ should NOT include type schemas (they go in describe_types)."""
        func = Function(func_with_enum)
        func_str = str(func)

        # Type schemas are now in PythonRuntime.describe_types(), not in Function.__str__()
        assert "types:" not in func_str
        # But Function still stores them internally
        assert "Status" in func.type_schemas
        assert "Priority" in func.type_schemas

    def test_function_can_disable_type_schemas(self):
        """Function with include_type_schemas=False should skip extraction."""
        func = Function(func_with_enum, include_type_schemas=False)

        assert func.type_schemas == {}

    def test_function_with_dataclass(self):
        """Function with dataclass parameter should extract schema."""
        func = Function(func_with_dataclass)

        assert "SimpleDataclass" in func.type_schemas
        assert "NestedDataclass" in func.type_schemas


# =============================================================================
# Tests - Variable Type Schema Control
# =============================================================================

class TestVariableTypeSchemaControl:
    """Test Variable include_type_schema and include_type_doc control."""

    def test_schema_true_doc_true(self):
        """include_type_schema=True, include_type_doc=True shows full schema with doc."""
        from cave_agent.python_runtime import PythonRuntime, Variable

        class MyClass:
            """My class docstring."""
            def my_method(self) -> str:
                return "test"

        runtime = PythonRuntime(variables=[
            Variable('obj', MyClass(), 'test', include_type_schema=True, include_type_doc=True),
        ])
        result = runtime.describe_types()

        assert "MyClass:" in result
        assert "doc: My class docstring." in result
        assert "methods:" in result
        assert "my_method()" in result

    def test_schema_true_doc_false(self):
        """include_type_schema=True, include_type_doc=False shows schema without doc."""
        from cave_agent.python_runtime import PythonRuntime, Variable

        class MyClass:
            """My class docstring."""
            def my_method(self) -> str:
                return "test"

        runtime = PythonRuntime(variables=[
            Variable('obj', MyClass(), 'test', include_type_schema=True, include_type_doc=False),
        ])
        result = runtime.describe_types()

        assert "MyClass:" in result
        assert "doc:" not in result
        assert "methods:" in result
        assert "my_method()" in result

    def test_schema_false_doc_true(self):
        """include_type_schema=False, include_type_doc=True shows doc only."""
        from cave_agent.python_runtime import PythonRuntime, Variable

        class MyClass:
            """My class docstring."""
            def my_method(self) -> str:
                return "test"

        runtime = PythonRuntime(variables=[
            Variable('obj', MyClass(), 'test', include_type_schema=False, include_type_doc=True),
        ])
        result = runtime.describe_types()

        assert "MyClass:" in result
        assert "doc: My class docstring." in result
        assert "methods:" not in result
        assert "my_method()" not in result

    def test_schema_false_doc_false(self):
        """include_type_schema=False, include_type_doc=False shows nothing."""
        from cave_agent.python_runtime import PythonRuntime, Variable

        class MyClass:
            """My class docstring."""
            def my_method(self) -> str:
                return "test"

        runtime = PythonRuntime(variables=[
            Variable('obj', MyClass(), 'test', include_type_schema=False, include_type_doc=False),
        ])
        result = runtime.describe_types()

        assert result == ""

    def test_any_true_wins_for_schema(self):
        """If ANY variable has include_type_schema=True, schema is shown."""
        from cave_agent.python_runtime import PythonRuntime, Variable

        class MyClass:
            """My class docstring."""
            def my_method(self) -> str:
                return "test"

        runtime = PythonRuntime(variables=[
            Variable('obj1', MyClass(), 'test1', include_type_schema=False, include_type_doc=False),
            Variable('obj2', MyClass(), 'test2', include_type_schema=True, include_type_doc=False),
        ])
        result = runtime.describe_types()

        assert "MyClass:" in result
        assert "methods:" in result
        assert "my_method()" in result

    def test_any_true_wins_for_doc(self):
        """If ANY variable has include_type_doc=True, doc is shown."""
        from cave_agent.python_runtime import PythonRuntime, Variable

        class MyClass:
            """My class docstring."""
            def my_method(self) -> str:
                return "test"

        runtime = PythonRuntime(variables=[
            Variable('obj1', MyClass(), 'test1', include_type_schema=True, include_type_doc=False),
            Variable('obj2', MyClass(), 'test2', include_type_schema=False, include_type_doc=True),
        ])
        result = runtime.describe_types()

        assert "MyClass:" in result
        assert "doc: My class docstring." in result
        assert "methods:" in result  # Because obj1 has schema=True

    def test_doc_only_no_class_doc(self):
        """Doc-only mode with no docstring returns nothing for that type."""
        from cave_agent.python_runtime import PythonRuntime, Variable

        class NoDocClass:
            def my_method(self) -> str:
                return "test"

        runtime = PythonRuntime(variables=[
            Variable('obj', NoDocClass(), 'test', include_type_schema=False, include_type_doc=True),
        ])
        result = runtime.describe_types()

        # No doc available, so nothing shown
        assert result == ""

    def test_multiple_types_mixed_settings(self):
        """Multiple types with different settings are handled correctly."""
        from cave_agent.python_runtime import PythonRuntime, Variable

        class TypeA:
            """TypeA doc."""
            def method_a(self) -> None:
                pass

        class TypeB:
            """TypeB doc."""
            def method_b(self) -> None:
                pass

        runtime = PythonRuntime(variables=[
            Variable('a', TypeA(), 'A', include_type_schema=True, include_type_doc=True),
            Variable('b', TypeB(), 'B', include_type_schema=False, include_type_doc=True),
        ])
        result = runtime.describe_types()

        # TypeA should have full schema
        assert "TypeA:" in result
        assert "method_a()" in result

        # TypeB should have doc only
        assert "TypeB:" in result
        assert "TypeB doc." in result
        assert "method_b()" not in result