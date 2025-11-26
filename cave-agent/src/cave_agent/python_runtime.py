from typing import (
    Callable, List, Dict, Any, Optional, Set, Union,
    get_args, get_origin, ForwardRef
)
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import inspect
from .security_checker import SecurityChecker, SecurityError
from traitlets.config import Config
from enum import Enum
from dataclasses import is_dataclass, fields as dataclass_fields, MISSING
from pydantic import BaseModel

class ExecutionResult:
    """
    Represents the result of code execution.
    """
    error: Optional[BaseException] = None
    stdout: Optional[str] = None

    def __init__(self, error: Optional[BaseException] = None, stdout: Optional[str] = None):
        self.error = error
        self.stdout = stdout

    @property
    def success(self):
        return self.error is None

class ErrorFeedbackMode(Enum):
    """Error feedback modes for LLM agent observation."""
    PLAIN = "Plain"      # Full traceback for agent debugging
    MINIMAL = "Minimal"     # Brief error info for agent efficiency

class PythonExecutor:
    """
    Handles Python code execution using IPython.
    """

    def __init__(self, security_checker: Optional[SecurityChecker] = None, error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN):
        """Initialize IPython shell for code execution."""
        ipython_config = self.create_ipython_config(error_feedback_mode=error_feedback_mode)
        self._shell = InteractiveShell(config=ipython_config)
        self._security_checker = security_checker

    def inject_into_namespace(self, name: str, value: Any):
        """Inject a value into the execution namespace."""
        self._shell.user_ns[name] = value
    
    async def execute(self, code: str) -> ExecutionResult:
        """Execute code snippet with optional security checks.
        
        Args:
            code: Python code to execute
            
        Returns:
            ExecutionResult with success status and output or error
        """   
        try:
            # Perform security check
            if self._security_checker:
                violations = self._security_checker.check_code(code)
                if len(violations) > 0:
                    violation_details = [str(v) for v in violations]
                    error_message = (
                        f"Code execution blocked: {len(violations)} violations found:\n"
                        + "\n".join(f"  - {detail}" for detail in violation_details)
                    )
                    security_error = SecurityError(error_message)
                    return ExecutionResult(error=security_error, stdout=None)
            
            # Execute the code
            with capture_output() as output:
                transformed_code = self._shell.transform_cell(code)
                result = await self._shell.run_cell_async(
                    transformed_code, 
                    transformed_cell=transformed_code
                )

            # Handle execution errors
            if result.error_before_exec:
                return ExecutionResult(
                    error=result.error_before_exec, 
                    stdout=output.stdout
                )
            if result.error_in_exec:
                return ExecutionResult(
                    error=result.error_in_exec, 
                    stdout=output.stdout
                )
            
            return ExecutionResult(stdout=output.stdout)

        except Exception as e:
            return ExecutionResult(error=e)
    
    def get_from_namespace(self, name: str) -> Any:
        """Get a value from the execution namespace."""
        return self._shell.user_ns.get(name)
    
    def reset(self):
        """Reset the shell"""
        self._shell.reset()
        import gc
        gc.collect()
        
    @staticmethod
    def create_ipython_config(error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN) -> Config:
        """Create a clean IPython configuration optimized for code execution."""
        config = Config()
        config.InteractiveShell.cache_size = 0 
        config.InteractiveShell.history_length = 0
        config.InteractiveShell.automagic = False
        config.InteractiveShell.separate_in = ''
        config.InteractiveShell.separate_out = ''
        config.InteractiveShell.separate_out2 = ''
        config.InteractiveShell.autocall = 0
        config.InteractiveShell.colors = 'nocolor'
        config.InteractiveShell.xmode = error_feedback_mode.value
        config.InteractiveShell.quiet = True
        config.InteractiveShell.autoindent = False
        
        return config


class Variable:
    """Represents a variable in the Python runtime environment."""

    name: str
    description: Optional[str]
    value: Optional[Any]
    type: str
    include_type_schema: bool
    include_type_doc: bool

    def __init__(
        self,
        name: str,
        value: Optional[Any] = None,
        description: Optional[str] = None,
        include_type_schema: bool = False,
        include_type_doc: bool = False,
    ):
        """
        Initialize the variable.

        Args:
            name: Variable name
            value: The value to store
            description: Optional description of the variable
            include_type_schema: Whether to include type methods/fields in types section
            include_type_doc: Whether to include type docstring in types section
        """
        self.name = name
        self.value = value
        self.description = description
        self.type = type(self.value).__name__ if self.value is not None else "NoneType"
        self.include_type_schema = include_type_schema
        self.include_type_doc = include_type_doc

    def __str__(self) -> str:
        """Return a string representation of the variable (without inline schema)."""
        parts = [f"- name: {self.name}"]
        parts.append(f"  type: {self.type}")
        if self.description:
            parts.append(f"  description: {self.description}")

        return "\n".join(parts)


class TypeSchemaExtractor:
    """
    Extracts and formats type schema information from Python types.

    Supports Pydantic models, dataclasses, and enums with recursive
    type discovery. Thread-safe through use of local state.
    """

    @classmethod
    def extract_from_instance(cls, value: Any, include_doc: bool = True) -> Optional[str]:
        """
        Extract type schema from a value/instance.

        For class instances, extracts either:
        - Field schema (for Pydantic models, dataclasses)
        - Method signatures (for regular classes)

        Args:
            value: The value/instance to analyze
            include_doc: Whether to include docstring in schema

        Returns:
            Formatted schema string or None for simple types
        """
        if value is None:
            return None

        value_type = type(value)

        # Skip built-in types
        if value_type in (str, int, float, bool, bytes, list, dict, tuple, set, frozenset):
            return None

        # Handle Enum instances
        if isinstance(value, Enum):
            return cls._format_enum_schema(value_type)

        # Handle Pydantic model instances
        if isinstance(value, BaseModel):
            processed: Set[str] = set()
            schemas: Dict[str, str] = {}
            return cls._format_pydantic_schema(value_type, processed, schemas, include_doc)

        # Handle dataclass instances
        if is_dataclass(value) and not isinstance(value, type):
            processed: Set[str] = set()
            schemas: Dict[str, str] = {}
            return cls._format_dataclass_schema(value_type, processed, schemas, include_doc)

        # Handle regular class instances - extract method signatures
        if hasattr(value_type, '__dict__'):
            return cls._format_class_methods(value_type, include_doc)

        return None

    @classmethod
    def _format_class_methods(cls, class_type: type, include_doc: bool = True) -> Optional[str]:
        """
        Extract public method signatures from a class.

        Args:
            class_type: The class to analyze
            include_doc: Whether to include docstring

        Returns:
            Formatted method signatures or None if no methods found
        """
        methods = []

        for name, method in inspect.getmembers(class_type, predicate=inspect.isfunction):
            # Skip private and magic methods
            if name.startswith('_'):
                continue

            try:
                sig = inspect.signature(method)
                # Format parameters (skip 'self')
                params = []
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    param_str = param_name
                    if param.annotation != inspect.Parameter.empty:
                        param_str += f": {cls._format_type_annotation(param.annotation)}"
                    if param.default != inspect.Parameter.empty:
                        param_str += f" = {param.default!r}"
                    params.append(param_str)

                # Format return type
                return_str = ""
                if sig.return_annotation != inspect.Signature.empty:
                    return_str = f" -> {cls._format_type_annotation(sig.return_annotation)}"

                method_sig = f"{name}({', '.join(params)}){return_str}"
                methods.append(f"    - {method_sig}")
            except (ValueError, TypeError):
                # Skip methods that can't be inspected
                continue

        if not methods:
            return None

        lines = [f"{class_type.__name__}:"]

        # Add docstring if available and requested
        if include_doc and class_type.__doc__ and class_type.__doc__.strip():
            lines.append(f"  doc: {class_type.__doc__.strip()}")

        lines.append("  methods:")
        lines.extend(methods)
        return "\n".join(lines)

    @classmethod
    def extract_from_signature(cls, func: Callable) -> Dict[str, str]:
        """
        Extract type schemas from a function signature.

        Args:
            func: Function to analyze

        Returns:
            Dictionary mapping type names to their schema strings
        """
        processed: Set[str] = set()
        schemas: Dict[str, str] = {}

        sig = inspect.signature(func)

        # Process parameter types
        for param in sig.parameters.values():
            if param.annotation != inspect.Parameter.empty:
                cls._process_type(param.annotation, processed, schemas)

        # Process return type
        if sig.return_annotation != inspect.Signature.empty:
            cls._process_type(sig.return_annotation, processed, schemas)

        return schemas

    @classmethod
    def _process_type(
        cls,
        type_hint: Any,
        processed: Set[str],
        schemas: Dict[str, str]
    ) -> None:
        """
        Recursively process a type hint to extract schemas.

        Args:
            type_hint: Type annotation to process
            processed: Set of already processed type names (prevents cycles)
            schemas: Dictionary to store extracted schemas
        """
        # Handle None type
        if type_hint is type(None) or type_hint is None:
            return

        # Handle ForwardRef (string annotations)
        if isinstance(type_hint, ForwardRef):
            # Cannot resolve forward refs without evaluation context
            return

        # Handle string annotations (forward references)
        if isinstance(type_hint, str):
            return

        # Get origin for generic types (List, Dict, Union, etc.)
        origin = get_origin(type_hint)

        # If it's a generic type, process its arguments
        if origin is not None:
            # Handle Union types (including Optional)
            args = get_args(type_hint)
            for arg in args:
                if arg is not type(None):  # Skip None in Optional types
                    cls._process_type(arg, processed, schemas)
            return

        # Skip built-in types
        if type_hint in (str, int, float, bool, bytes, list, dict, tuple, set, frozenset, type):
            return

        # Handle Callable - just skip, too complex to schema
        if type_hint is Callable:
            return

        # Get type name
        if not hasattr(type_hint, '__name__'):
            return

        type_name = type_hint.__name__
        if type_name in processed:
            return

        processed.add(type_name)

        # Extract schema for custom types
        schema = cls._extract_schema(type_hint, processed, schemas)
        if schema:
            schemas[type_name] = schema

    @classmethod
    def _extract_schema(
        cls,
        type_hint: Any,
        processed: Set[str],
        schemas: Dict[str, str],
        include_doc: bool = True
    ) -> Optional[str]:
        """
        Extract schema information from a type.

        Args:
            type_hint: Type to extract schema from
            processed: Set of already processed type names
            schemas: Dictionary to store extracted schemas
            include_doc: Whether to include docstring

        Returns:
            Formatted schema string or None
        """
        # Handle Pydantic models
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return cls._format_pydantic_schema(type_hint, processed, schemas, include_doc)

        # Handle dataclasses
        if is_dataclass(type_hint) and isinstance(type_hint, type):
            return cls._format_dataclass_schema(type_hint, processed, schemas, include_doc)

        # Handle Enums
        if inspect.isclass(type_hint) and issubclass(type_hint, Enum):
            return cls._format_enum_schema(type_hint)

        return None

    @classmethod
    def _format_pydantic_schema(
        cls,
        model: type,
        processed: Set[str],
        schemas: Dict[str, str],
        include_doc: bool = True
    ) -> str:
        """
        Format a Pydantic model schema.

        Args:
            model: Pydantic model class
            processed: Set of already processed type names
            schemas: Dictionary to store extracted schemas
            include_doc: Whether to include docstring

        Returns:
            Formatted schema string
        """
        lines = [f"{model.__name__}:"]

        # Add docstring if available and requested
        if include_doc and model.__doc__ and model.__doc__.strip():
            lines.append(f"  doc: {model.__doc__.strip()}")

        lines.append("  fields:")

        for field_name, field_info in model.model_fields.items():
            field_type = field_info.annotation
            type_str = cls._format_type_annotation(field_type)

            field_line = f"    - {field_name}: {type_str}"

            if field_info.description:
                field_line += f"  # {field_info.description}"

            lines.append(field_line)

            # Recursively process field types
            cls._process_type(field_type, processed, schemas)

        return "\n".join(lines)

    @classmethod
    def _format_dataclass_schema(
        cls,
        dataclass_type: type,
        processed: Set[str],
        schemas: Dict[str, str],
        include_doc: bool = True
    ) -> str:
        """
        Format a dataclass schema.

        Args:
            dataclass_type: Dataclass type
            processed: Set of already processed type names
            schemas: Dictionary to store extracted schemas
            include_doc: Whether to include docstring

        Returns:
            Formatted schema string
        """
        lines = [f"{dataclass_type.__name__}:"]

        # Add docstring if available and requested
        if include_doc and dataclass_type.__doc__ and dataclass_type.__doc__.strip():
            lines.append(f"  doc: {dataclass_type.__doc__.strip()}")

        lines.append("  fields:")

        for field in dataclass_fields(dataclass_type):
            type_str = cls._format_type_annotation(field.type)
            field_line = f"    - {field.name}: {type_str}"

            if field.default is not MISSING:
                field_line += f" = {field.default!r}"
            elif field.default_factory is not MISSING:  # type: ignore
                field_line += " = <factory>"

            lines.append(field_line)

            # Recursively process field types
            cls._process_type(field.type, processed, schemas)

        return "\n".join(lines)

    @classmethod
    def _format_enum_schema(cls, enum_type: type) -> str:
        """
        Format an Enum schema.

        Args:
            enum_type: Enum class

        Returns:
            Formatted schema string
        """
        lines = [f"{enum_type.__name__} (Enum):"]
        for member in enum_type:
            lines.append(f"  - {member.name} = {member.value!r}")
        return "\n".join(lines)

    @classmethod
    def _format_type_annotation(cls, type_hint: Any) -> str:
        """
        Format a type annotation as a readable string.

        Args:
            type_hint: Type annotation to format

        Returns:
            Human-readable type string
        """
        if type_hint is type(None) or type_hint is None:
            return "None"

        # Handle ForwardRef
        if isinstance(type_hint, ForwardRef):
            return type_hint.__forward_arg__

        # Handle string annotations
        if isinstance(type_hint, str):
            return type_hint

        origin = get_origin(type_hint)

        # Handle generic types
        if origin is not None:
            args = get_args(type_hint)

            # Get a readable origin name
            if origin is Union:
                # Special handling for Optional (Union[X, None])
                non_none_args = [a for a in args if a is not type(None)]
                if len(non_none_args) == 1 and len(args) == 2:
                    return f"Optional[{cls._format_type_annotation(non_none_args[0])}]"
                arg_strs = [cls._format_type_annotation(arg) for arg in args]
                return f"Union[{', '.join(arg_strs)}]"

            origin_name = getattr(origin, '__name__', str(origin).replace('typing.', ''))

            if not args:
                return origin_name

            arg_strs = [cls._format_type_annotation(arg) for arg in args]
            return f"{origin_name}[{', '.join(arg_strs)}]"

        # Handle Callable without arguments
        if type_hint is Callable:
            return "Callable"

        # Return the type name
        if hasattr(type_hint, '__name__'):
            return type_hint.__name__

        return str(type_hint)


class Function:
    """
    Represents a function in the Python runtime environment.
    """

    def __init__(
        self,
        func: Callable,
        description: Optional[str] = None,
        include_doc: bool = True,
        include_type_schemas: bool = True,
    ) -> None:
        """
        Initialize the function.
        
        Args:
            func: Callable function to wrap
            description: Optional description of the function
            include_doc: Whether to include the function's docstring
            include_type_schemas: Whether to extract type schemas
        """
        self.func = func
        self.description = description
        self.name = func.__name__
        self.signature = f"{self.name}{inspect.signature(func)}"
        self.doc: Optional[str] = None
        self.type_schemas: Dict[str, str] = {}

        if include_doc and hasattr(func, "__doc__") and func.__doc__:
            self.doc = func.__doc__.strip()

        if include_type_schemas:
            self.type_schemas = TypeSchemaExtractor.extract_from_signature(func)
    
    def __str__(self) -> str:
        """Return a string representation of the function."""
        parts = [f"- function: {self.signature}"]
        
        if self.description:
            parts.append(f"  description: {self.description}")

        if self.doc:
            parts.append(self._format_docstring())

        return "\n".join(parts)

    def _format_docstring(self) -> str:
        """
        Format the docstring with proper indentation.

        Returns:
            Formatted docstring
        """
        lines = ["  doc:"]
        for line in self.doc.split('\n'):
            lines.append(f"    {line}")
        return "\n".join(lines)
    
class PythonRuntime:
    """
    A Python runtime that executes code snippets in an IPython environment.
    Provides a controlled execution environment with registered functions and objects.
    """

    _executor: PythonExecutor
    _functions: Dict[str, Function]
    _variables: Dict[str, Variable]

    def __init__(
        self,
        functions: Optional[List[Function]] = None,
        variables: Optional[List[Variable]] = None,
        security_checker: Optional[SecurityChecker] = None,
        error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN,
    ):
        """
        Initialize runtime with executor and optional initial resources.

        Args:
            functions: List of functions to inject into runtime
            variables: List of variables to inject into runtime
            security_checker: Security checker instance to use for code execution
            error_feedback_mode: Error feedback mode for execution errors
        """
        self._executor = PythonExecutor(security_checker=security_checker, error_feedback_mode=error_feedback_mode)
        self._functions = {}
        self._variables = {}

        for function in (functions or []):
            self.inject_function(function)

        for variable in (variables or []):
            self.inject_variable(variable)

    def inject_function(self, function: Function):
        """Inject a function in both metadata and execution namespace."""
        if function.name in self._functions:
            raise ValueError(f"Function '{function.name}' already exists")
        self._functions[function.name] = function
        self._executor.inject_into_namespace(function.name, function.func)
    
    def inject_variable(self, variable: Variable):
        """Inject a variable in both metadata and execution namespace."""
        if variable.name in self._variables:
            raise ValueError(f"Variable '{variable.name}' already exists")
        self._variables[variable.name] = variable
        self._executor.inject_into_namespace(variable.name, variable.value)

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code using the executor."""
        return await self._executor.execute(code)

    def get_variable_value(self, name: str) -> Any:
        """Get current value of a variable."""
        if name not in self._variables:
            raise KeyError(f"Variable '{name}' is not managed by this runtime. Available variables: {list(self._variables.keys())}")
        return self._executor.get_from_namespace(name)
    
    def describe_variables(self) -> str:
        """Generate formatted variable descriptions for system prompt."""
        if not self._variables:
            return "No variables available"

        descriptions = []
        for variable in self._variables.values():
            descriptions.append(str(variable))

        return "\n".join(descriptions)

    def describe_functions(self) -> str:
        """Generate formatted function descriptions for system prompt."""
        if not self._functions:
            return "No functions available"

        descriptions = []
        for function in self._functions.values():
            descriptions.append(str(function))
        
        return "\n".join(descriptions)

    def describe_types(self) -> str:
        """
        Generate deduplicated type schemas from both functions and variables.

        Collects all unique type schemas and returns them in a single section.
        - If ANY variable of a type has include_type_schema=True, show full schema
        - If ANY variable of a type has include_type_doc=True, show doc
        - If include_type_schema=False but include_type_doc=True, show doc only
        """
        type_schemas: Dict[str, str] = {}
        processed_types: Set[str] = set()

        # Group variables by type and determine settings
        # ANY True wins for both include_type_schema and include_type_doc
        type_include_schema: Dict[str, bool] = {}
        type_include_doc: Dict[str, bool] = {}
        type_sample_value: Dict[str, Any] = {}

        for variable in self._variables.values():
            if variable.value is None:
                continue

            # Skip if both are False
            if not variable.include_type_schema and not variable.include_type_doc:
                continue

            type_name = variable.type

            # Store sample value for schema generation
            if type_name not in type_sample_value:
                type_sample_value[type_name] = variable.value
                type_include_schema[type_name] = variable.include_type_schema
                type_include_doc[type_name] = variable.include_type_doc
            else:
                # If ANY variable wants schema, include it
                if variable.include_type_schema:
                    type_include_schema[type_name] = True
                # If ANY variable wants doc, include it
                if variable.include_type_doc:
                    type_include_doc[type_name] = True

        # Generate schemas for variable types
        for type_name, value in type_sample_value.items():
            include_schema = type_include_schema.get(type_name, False)
            include_doc = type_include_doc.get(type_name, False)

            if include_schema:
                # Full schema with/without doc
                schema = TypeSchemaExtractor.extract_from_instance(value, include_doc)
            elif include_doc:
                # Doc only (no methods/fields)
                schema = self._format_doc_only(type_name, value)
            else:
                schema = None

            if schema:
                type_schemas[type_name] = schema
                processed_types.add(type_name)

        # Collect from functions (include doc by default)
        for function in self._functions.values():
            for type_name, schema in function.type_schemas.items():
                if type_name not in processed_types:
                    type_schemas[type_name] = schema
                    processed_types.add(type_name)

        if not type_schemas:
            return ""

        lines = []
        for schema in type_schemas.values():
            lines.append(schema)

        return "\n".join(lines)

    def _format_doc_only(self, type_name: str, value: Any) -> Optional[str]:
        """Format type with doc only (no methods/fields)."""
        value_type = type(value)
        if value_type.__doc__ and value_type.__doc__.strip():
            return f"{type_name}:\n  doc: {value_type.__doc__.strip()}"
        return None

    def reset(self):
        """Reset the runtime."""
        self._executor.reset()
        self._functions.clear()
        self._variables.clear()