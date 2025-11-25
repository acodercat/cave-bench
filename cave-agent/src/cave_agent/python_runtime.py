from typing import (
    Callable, List, Dict, Any, Optional, Set, Union,
    get_args, get_origin, ForwardRef, TYPE_CHECKING
)
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import inspect
from .security_checker import SecurityChecker, SecurityError
from traitlets.config import Config
from enum import Enum
from dataclasses import is_dataclass, fields as dataclass_fields, MISSING

# Lazy import for optional Pydantic support
try:
    from pydantic import BaseModel as PydanticBaseModel
    HAS_PYDANTIC = True
except ImportError:
    PydanticBaseModel = None  # type: ignore
    HAS_PYDANTIC = False

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
            
        except SecurityError:
            # Re-raise security errors as-is
            raise
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
        """"Create a clean IPython configuration optimized for code execution."""
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
    description: Optional[str] = None
    value: Optional[Any] = None
    doc: Optional[str] = None
    type: str

    def __init__(self, name: str, value: Optional[Any] = None, description: Optional[str] = None, include_doc: bool = True):
        """Initialize the variable."""
        self.name = name
        self.value = value
        self.description = description
        self.type = type(self.value).__name__

        if include_doc and hasattr(self.value, "__doc__") and self.value.__doc__ and self.value.__doc__.strip():
            self.doc = self.value.__doc__.strip()
        
    def __str__(self):
        """Return a string representation of the variable."""
        parts = [f"- name: {self.name}"]
        parts.append(f"  type: {self.type}")
        if self.description:
            parts.append(f"  description: {self.description}")
        if self.doc:
            parts.append(f"  doc: {self.doc}")

        return "\n".join(parts)


class TypeSchemaExtractor:
    """
    Extracts and formats type schema information from Python types.

    Supports Pydantic models, dataclasses, and enums with recursive
    type discovery. Thread-safe through use of local state.
    """

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
        schemas: Dict[str, str]
    ) -> Optional[str]:
        """
        Extract schema information from a type.

        Args:
            type_hint: Type to extract schema from
            processed: Set of already processed type names
            schemas: Dictionary to store extracted schemas

        Returns:
            Formatted schema string or None
        """
        # Handle Pydantic models (only if pydantic is available)
        if HAS_PYDANTIC and isinstance(type_hint, type) and issubclass(type_hint, PydanticBaseModel):
            return cls._format_pydantic_schema(type_hint, processed, schemas)

        # Handle dataclasses
        if is_dataclass(type_hint) and isinstance(type_hint, type):
            return cls._format_dataclass_schema(type_hint, processed, schemas)

        # Handle Enums
        if inspect.isclass(type_hint) and issubclass(type_hint, Enum):
            return cls._format_enum_schema(type_hint)

        return None

    @classmethod
    def _format_pydantic_schema(
        cls,
        model: type,
        processed: Set[str],
        schemas: Dict[str, str]
    ) -> str:
        """
        Format a Pydantic model schema.

        Args:
            model: Pydantic model class
            processed: Set of already processed type names
            schemas: Dictionary to store extracted schemas

        Returns:
            Formatted schema string
        """
        lines = [f"{model.__name__}:"]

        # Support both Pydantic v1 and v2
        if hasattr(model, 'model_fields'):
            # Pydantic v2
            for field_name, field_info in model.model_fields.items():
                field_type = field_info.annotation
                type_str = cls._format_type_annotation(field_type)

                field_line = f"  - {field_name}: {type_str}"

                if field_info.description:
                    field_line += f"  # {field_info.description}"

                lines.append(field_line)

                # Recursively process field types
                cls._process_type(field_type, processed, schemas)
        elif hasattr(model, '__fields__'):
            # Pydantic v1
            for field_name, field_info in model.__fields__.items():
                field_type = field_info.outer_type_
                type_str = cls._format_type_annotation(field_type)

                field_line = f"  - {field_name}: {type_str}"

                if field_info.field_info.description:
                    field_line += f"  # {field_info.field_info.description}"

                lines.append(field_line)

                # Recursively process field types
                cls._process_type(field_type, processed, schemas)

        return "\n".join(lines)

    @classmethod
    def _format_dataclass_schema(
        cls,
        dataclass_type: type,
        processed: Set[str],
        schemas: Dict[str, str]
    ) -> str:
        """
        Format a dataclass schema.

        Args:
            dataclass_type: Dataclass type
            processed: Set of already processed type names
            schemas: Dictionary to store extracted schemas

        Returns:
            Formatted schema string
        """
        lines = [f"{dataclass_type.__name__}:"]

        for field in dataclass_fields(dataclass_type):
            type_str = cls._format_type_annotation(field.type)
            field_line = f"  - {field.name}: {type_str}"

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
        
        if self.type_schemas:
            parts.append(self._format_type_schemas())
        
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
    
    def _format_type_schemas(self) -> str:
        """
        Format type schemas with proper indentation.
        
        Returns:
            Formatted type schemas
        """
        lines = ["  types:"]
        for type_name, schema in self.type_schemas.items():
            for line in schema.split('\n'):
                lines.append(f"    {line}")
        return "\n".join(lines)
    
class PythonRuntime:
    """
    A Python runtime that executes code snippets in an IPython environment.
    Provides a controlled execution environment with registered functions and objects.
    """
    def __init__(
        self,
        functions: List[Function] = [],
        variables: List[Variable] = [],
        security_checker: Optional[SecurityChecker] = None,
        error_feedback_mode: ErrorFeedbackMode = ErrorFeedbackMode.PLAIN,
    ):
        """
        Initialize runtime with executor and optional initial resources.
        
        Args:
            functions: List of functions to inject into runtime
            variables: List of variables to inject into runtime
            security_checker: Security checker instance to use for code execution
        """
            
        self._executor = PythonExecutor(security_checker=security_checker, error_feedback_mode=error_feedback_mode)
        self._functions: Dict[str, Function] = {}
        self._variables: Dict[str, Variable] = {}

        for function in functions:
            self.inject_function(function)
        
        for variable in variables:
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
    
    def reset(self):
        """Reset the runtime."""
        self._executor.reset()
        self._functions.clear()
        self._variables.clear()