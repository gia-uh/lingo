from pydantic import BaseModel
from typing import get_type_hints, Union, Any


def type_to_str(tp: Any) -> str:
    """Convert a type to a string representation."""
    # Handle Optional types (Union[..., None])
    origin = getattr(tp, "__origin__", None)
    args = getattr(tp, "__args__", ())

    if origin is Union and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return f"Optional[{type_to_str(non_none[0])}]"

    # Handle basic types
    if hasattr(tp, "__name__"):
        return tp.__name__

    # Handle generic types like List, Dict
    if hasattr(tp, "_name"):  # for types like List, Dict from typing
        args_str = ", ".join(type_to_str(a) for a in args)
        return f"{tp._name}[{args_str}]"

    # Fallback
    return str(tp).replace("typing.", "")


def generate_pydantic_code(model_cls: type[BaseModel]) -> str:
    """
    Generate Python source code for a Pydantic BaseModel subclass.
    """
    lines: list[str] = []
    visited: set[str] = set()

    def generate(cls: type[BaseModel], lines: list[str], visited: set[str]):
        class_name = cls.__name__

        if class_name in visited:
            return

        visited.add(class_name)

        # Generate the class definition
        class_lines = [f"class {class_name}(BaseModel):"]

        subtypes = []

        # Get type hints (annotations) for the model
        try:
            hints = get_type_hints(cls)
        except (AttributeError, TypeError, NameError):
            # Fallback for complex/unresolvable types
            hints = {}

        if not cls.model_fields:
            class_lines.append("    pass")

        # For each field, get name, type, and default value if any
        for field_name, field in cls.model_fields.items():
            field_type = hints.get(field_name, Any)
            type_str = type_to_str(field_type)

            # Determine default value
            if not field.is_required():
                default_val = repr(field.default)
                line = f"    {field_name}: {type_str} = {default_val}"
            else:
                line = f"    {field_name}: {type_str}"

            class_lines.append(line)

            # Recurse for nested Pydantic models
            origin_type = getattr(field_type, "__origin__", None)
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                subtypes.append(field_type)
            elif origin_type:
                # Check args for nested models
                for arg in getattr(field_type, "__args__", []):
                    if isinstance(arg, type) and issubclass(arg, BaseModel):
                        subtypes.append(arg)

        # Add the class lines to the main lines
        lines.extend(class_lines)
        lines.append("")  # Add a blank line

        # Generate subtypes after the current class
        for sub_cls in subtypes:
            if sub_cls.__name__ not in visited:
                generate(sub_cls, lines, visited)

    generate(model_cls, lines, visited)
    return "\n".join(lines)


def tee(*functions):
    """
    Calls all functions with the same argunents.
    """

    def wrapper(*args, **kwargs):
        for fn in functions:
            fn(*args, **kwargs)

    return wrapper
