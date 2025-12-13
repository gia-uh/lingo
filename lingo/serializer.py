import yaml
import inspect
from typing import Any, Dict, Type, Optional, Union
from pydantic import BaseModel
import importlib
from .flow_validator import ValidationError, FlowValidator
from .llm import Message
from .tools import Tool, DelegateTool
from .flow import (
    Node, Append, Prepend, Reply, Invoke, NoOp, Create,
    Sequence, Decide, Choose, Route, Flow, FunctionalNode
)

class FlowSerializer:
    """
    Handles serialization and deserialization of Flow objects to/from YAML.
    """
    
    def __init__(self, tool_registry: Optional[Dict[str, Tool]] = None):
        """
        Initialize the serializer with optional tool registry.
        
        Args:
            tool_registry: Dictionary mapping tool names to Tool instances
        """
        self.tool_registry = tool_registry or {}
        self._custom_classes = {}
        self.validator = FlowValidator()
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool instance for serialization/deserialization.
        
        Args:
            tool: Tool instance
        """
        self.tool_registry[tool.name] = tool
    
    def register_tool_class(self, tool_class: Type[Tool], **kwargs) -> None:
        """
        Register a tool class and instantiate it.
        
        Args:
            tool_class: Tool class to instantiate
            **kwargs: Arguments for the tool constructor
        """
        try:
            instance = tool_class(**kwargs)
            self.register_tool(instance)
        except Exception as e:
            print(f"Warning: Could not instantiate {tool_class.__name__}: {e}")
    
    def register_custom_class(self, class_path: str, alias: Optional[str] = None) -> None:
        """
        Register a custom class by import path.
        
        Args:
            class_path: Full import path (e.g., "myapp.models.User")
            alias: Optional alias for the class
        """
        module_name, class_name = class_path.rsplit('.', 1)
        self._custom_classes[alias or class_name] = (module_name, class_name)
    
    def validate_yaml(self, yaml_str: str) -> bool:
        """
        Validate YAML string without deserializing.
        
        Args:
            yaml_str: YAML string to validate
            
        Returns:
            True if valid, raises ValidationError if not
        """
        try:
            data = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML syntax: {e}")
        
        errors = self.validator.validate_yaml(data)
        if errors:
            raise ValidationError("YAML validation failed", errors)
        
        return True
    
    def validate_yaml_file(self, filepath: str) -> bool:
        """
        Validate YAML file without loading.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            True if valid, raises ValidationError if not
        """
        with open(filepath, 'r') as f:
            yaml_str = f.read()
        
        return self.validate_yaml(yaml_str)
    
    def serialize_flow(self, flow: Flow) -> Dict[str, Any]:
        """
        Serialize a Flow object to a dictionary.
        
        Args:
            flow: Flow object to serialize
            
        Returns:
            Dictionary representation
        """
        return {
            "name": flow.name,
            "description": flow.description,
            "version": "1.0",
            "nodes": [self._serialize_node(node) for node in flow.nodes]
        }
    
    def _serialize_node(self, node: Node) -> Dict[str, Any]:
        """
        Serialize a single node to dictionary.
        
        Args:
            node: Node object
            
        Returns:
            Dictionary representation
        """
        node_type = type(node).__name__
        
        if isinstance(node, Append):
            return {
                "type": "append",
                "message": self._serialize_message(node.msg)
            }
        elif isinstance(node, Prepend):
            return {
                "type": "prepend",
                "message": self._serialize_message(node.msg)
            }
        elif isinstance(node, Reply):
            return {
                "type": "reply",
                "instructions": [self._serialize_instruction(inst) for inst in node.instructions]
            }
        elif isinstance(node, Invoke):
            return {
                "type": "invoke",
                "tools": [self._serialize_tool(tool) for tool in node.tools]
            }
        elif isinstance(node, NoOp):
            return {"type": "noop"}
        elif isinstance(node, Create):
            return {
                "type": "create",
                "model": self._serialize_model(node.model),
                "instructions": [self._serialize_instruction(inst) for inst in node.instructions]
            }
        elif isinstance(node, Sequence):
            return {
                "type": "sequence",
                "nodes": [self._serialize_node(child) for child in node.nodes]
            }
        elif isinstance(node, Decide):
            return {
                "type": "decide",
                "on_true": self._serialize_node(node.on_true),
                "on_false": self._serialize_node(node.on_false),
                "instructions": [self._serialize_instruction(inst) for inst in node.instructions]
            }
        elif isinstance(node, Choose):
            return {
                "type": "choose",
                "choices": {
                    key: self._serialize_node(value) 
                    for key, value in node.choices.items()
                },
                "instructions": [self._serialize_instruction(inst) for inst in node.instructions]
            }
        elif isinstance(node, Route):
            return {
                "type": "route",
                "flows": [self.serialize_flow(flow) for flow in node.flows]
            }
        elif isinstance(node, FunctionalNode):
            raise ValueError(
                "FunctionalNode cannot be serialized. "
                "Use YAML for declarative flows only."
            )
        else:
            raise TypeError(f"Unsupported node type: {node_type}")
    
    def _serialize_message(self, message: Message) -> Dict[str, Any]:
        """
        Serialize a Message object.
        
        Args:
            message: Message object
            
        Returns:
            Dictionary representation
        """
        content = message.content
        
        # Handle different content types
        if isinstance(content, BaseModel):
            content_data = {
                "type": "pydantic_model",
                "model": f"{type(content).__module__}.{type(content).__name__}",
                "data": content.model_dump()
            }
        elif isinstance(content, dict):
            content_data = {
                "type": "dict",
                "value": content
            }
        elif isinstance(content, list):
            content_data = {
                "type": "list",
                "value": content
            }
        else:
            content_data = {
                "type": "string",
                "value": str(content)
            }
        
        return {
            "role": message.role,
            "content": content_data
        }
    
    def _serialize_instruction(self, instruction: Union[str, Message]) -> Dict[str, Any]:
        """
        Serialize an instruction.
        
        Args:
            instruction: String or Message
            
        Returns:
            Dictionary representation
        """
        if isinstance(instruction, Message):
            return {
                "type": "message",
                "value": self._serialize_message(instruction)
            }
        else:
            return {
                "type": "string",
                "value": instruction
            }
    
    def _serialize_tool(self, tool: Tool) -> Dict[str, Any]:
        """
        Serialize a Tool object.
        
        Args:
            tool: Tool object
            
        Returns:
            Dictionary representation
        """
        # Check if tool is in registry
        for name, registered_tool in self.tool_registry.items():
            if registered_tool is tool:
                return {
                    "type": "registered",
                    "name": tool.name
                }
        
        # Tool not in registry, serialize based on type
        if isinstance(tool, DelegateTool):
            # For DelegateTool, we need to handle the target function
            target = tool._target
            return {
                "type": "delegate_tool",
                "name": tool.name,
                "description": tool.description,
                "target_module": target.__module__,
                "target_name": target.__name__,
                "is_async": inspect.iscoroutinefunction(target)
            }
        else:
            # Regular tool
            return {
                "type": "custom_tool",
                "name": tool.name,
                "description": tool.description,
                "class": f"{type(tool).__module__}.{type(tool).__name__}"
            }
    
    def _serialize_model(self, model_class: Type[BaseModel]) -> Dict[str, Any]:
        """
        Serialize a Pydantic model class.
        
        Args:
            model_class: Pydantic model class
            
        Returns:
            Dictionary representation
        """
        return {
            "module": model_class.__module__,
            "name": model_class.__name__,
            "schema": model_class.model_json_schema()
        }
    
    def to_yaml(self, flow: Flow) -> str:
        """
        Convert Flow to YAML string.
        
        Args:
            flow: Flow object
            
        Returns:
            YAML string
        """
        flow_dict = self.serialize_flow(flow)
        return yaml.dump(flow_dict, default_flow_style=False, sort_keys=False, indent=2)
    
    def save_yaml(self, flow: Flow, filepath: str) -> None:
        """
        Save Flow to YAML file.
        
        Args:
            flow: Flow object
            filepath: Path to save file
        """
        yaml_str = self.to_yaml(flow)
        with open(filepath, 'w') as f:
            f.write(yaml_str)
    
    def deserialize_flow(self, flow_dict: Dict[str, Any]) -> Flow:
        """
        Deserialize a dictionary to Flow object.
        
        Args:
            flow_dict: Dictionary representation
            
        Returns:
            Flow object
        """
        # Validate before deserializing
        errors = self.validator.validate_yaml(flow_dict)
        if errors:
            raise ValidationError("Invalid flow structure", errors)
        
        flow = Flow(
            name=flow_dict.get("name", "UnnamedFlow"),
            description=flow_dict.get("description", "")
        )
        
        # Clear initial nodes
        flow.nodes = []
        
        # Reconstruct nodes
        for node_dict in flow_dict.get("nodes", []):
            node = self._deserialize_node(node_dict)
            if node:
                flow.nodes.append(node)
        
        return flow
    
    def _deserialize_node(self, node_dict: Dict[str, Any]) -> Node:
        """
        Deserialize a node dictionary.
        
        Args:
            node_dict: Node dictionary
            
        Returns:
            Node object
        """
        node_type = node_dict.get("type")
        
        if node_type == "append":
            return Append(self._deserialize_message(node_dict["message"]))
        elif node_type == "prepend":
            return Prepend(self._deserialize_message(node_dict["message"]))
        elif node_type == "reply":
            instructions = [
                self._deserialize_instruction(inst_dict)
                for inst_dict in node_dict.get("instructions", [])
            ]
            return Reply(*instructions)
        elif node_type == "invoke":
            tools = [
                self._deserialize_tool(tool_dict)
                for tool_dict in node_dict.get("tools", [])
            ]
            return Invoke(*tools)
        elif node_type == "noop":
            return NoOp()
        elif node_type == "create":
            model_class = self._deserialize_model(node_dict["model"])
            instructions = [
                self._deserialize_instruction(inst_dict)
                for inst_dict in node_dict.get("instructions", [])
            ]
            return Create(model_class, *instructions)
        elif node_type == "sequence":
            nodes = [
                self._deserialize_node(child_dict)
                for child_dict in node_dict.get("nodes", [])
            ]
            return Sequence(*nodes)
        elif node_type == "decide":
            on_true = self._deserialize_node(node_dict["on_true"])
            on_false = self._deserialize_node(node_dict["on_false"])
            instructions = [
                self._deserialize_instruction(inst_dict)
                for inst_dict in node_dict.get("instructions", [])
            ]
            return Decide(on_true, on_false, *instructions)
        elif node_type == "choose":
            choices = {
                key: self._deserialize_node(value_dict)
                for key, value_dict in node_dict.get("choices", {}).items()
            }
            instructions = [
                self._deserialize_instruction(inst_dict)
                for inst_dict in node_dict.get("instructions", [])
            ]
            return Choose(choices, *instructions)
        elif node_type == "route":
            flows = [
                self.deserialize_flow(flow_dict)
                for flow_dict in node_dict.get("flows", [])
            ]
            return Route(*flows)
        else:
            raise ValueError(f"Unknown node type: {node_type}")
    
    def _deserialize_message(self, msg_dict: Dict[str, Any]) -> Message:
        """
        Deserialize a message dictionary.
        
        Args:
            msg_dict: Message dictionary
            
        Returns:
            Message object
        """
        role = msg_dict["role"]
        content_data = msg_dict["content"]
        
        content_type = content_data.get("type", "string")
        
        if content_type == "pydantic_model":
            # Import and instantiate model
            model_path = content_data["model"]
            model_class = self._import_class(model_path)
            content = model_class(**content_data["data"])
        elif content_type == "dict":
            content = content_data["value"]
        elif content_type == "list":
            content = content_data["value"]
        else:
            content = content_data["value"]
        
        return Message(role=role, content=content)
    
    def _deserialize_instruction(self, inst_dict: Dict[str, Any]) -> Union[str, Message]:
        """
        Deserialize an instruction.
        
        Args:
            inst_dict: Instruction dictionary
            
        Returns:
            String or Message
        """
        if inst_dict["type"] == "message":
            return self._deserialize_message(inst_dict["value"])
        else:
            return inst_dict["value"]
    
    def _deserialize_tool(self, tool_dict: Dict[str, Any]) -> Tool:
        """
        Deserialize a tool.
        
        Args:
            tool_dict: Tool dictionary
            
        Returns:
            Tool object
        """
        tool_type = tool_dict.get("type", "custom_tool")
        
        if tool_type == "registered":
            tool_name = tool_dict["name"]
            if tool_name in self.tool_registry:
                return self.tool_registry[tool_name]
            else:
                raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        elif tool_type == "delegate_tool":
            # Handle DelegateTool with function target
            from .tools import DelegateTool
            
            # Import the target function
            module_name = tool_dict["target_module"]
            function_name = tool_dict["target_name"]
            
            try:
                module = importlib.import_module(module_name)
                target_func = getattr(module, function_name)
                
                # Create DelegateTool
                return DelegateTool(
                    name=tool_dict["name"],
                    description=tool_dict["description"],
                    target=target_func
                )
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Cannot import function {function_name} from {module_name}: {e}"
                )
        
        else:  # custom_tool
            # Custom tool - need to import class
            class_path = tool_dict["class"]
            tool_class = self._import_class(class_path)
            
            # Try to instantiate
            try:
                return tool_class(
                    name=tool_dict["name"],
                    description=tool_dict["description"]
                )
            except TypeError as e:
                # Try without parameters
                try:
                    return tool_class()
                except Exception:
                    raise TypeError(
                        f"Cannot instantiate tool {tool_dict['name']} "
                        f"from class {class_path}: {e}"
                    )
    
    def _deserialize_model(self, model_dict: Dict[str, Any]) -> Type[BaseModel]:
        """
        Deserialize a Pydantic model class.
        
        Args:
            model_dict: Model dictionary
            
        Returns:
            Pydantic model class
        """
        module_name = model_dict["module"]
        class_name = model_dict["name"]
        
        return self._import_class(f"{module_name}.{class_name}")
    
    def _import_class(self, class_path: str) -> Type:
        """
        Import a class from full path.
        
        Args:
            class_path: Full class path (e.g., "module.submodule.ClassName")
            
        Returns:
            Class object
        """
        module_name, class_name = class_path.rsplit('.', 1)
        
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            # Check custom classes
            if class_name in self._custom_classes:
                alt_module, alt_class = self._custom_classes[class_name]
                try:
                    module = importlib.import_module(alt_module)
                    return getattr(module, alt_class)
                except (ImportError, AttributeError):
                    pass
            
            raise ImportError(f"Cannot import {class_path}: {e}")
    
    def from_yaml(self, yaml_str: str) -> Flow:
        """
        Create Flow from YAML string.
        
        Args:
            yaml_str: YAML string
            
        Returns:
            Flow object
        """
        try:
            flow_dict = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML syntax: {e}")
        
        # Validate before deserializing
        errors = self.validator.validate_yaml(flow_dict)
        if errors:
            raise ValidationError("YAML validation failed", errors)
        
        return self.deserialize_flow(flow_dict)
    
    def load_yaml(self, filepath: str) -> Flow:
        """
        Load Flow from YAML file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            Flow object
        """
        with open(filepath, 'r') as f:
            yaml_str = f.read()
        
        return self.from_yaml(yaml_str)


# Convenience functions
_default_serializer = FlowSerializer()

def flow_to_yaml(flow: Flow) -> str:
    """Convert Flow to YAML string using default serializer."""
    return _default_serializer.to_yaml(flow)

def flow_from_yaml(yaml_str: str) -> Flow:
    """Create Flow from YAML string using default serializer."""
    return _default_serializer.from_yaml(yaml_str)

def save_flow_yaml(flow: Flow, filepath: str) -> None:
    """Save Flow to YAML file."""
    _default_serializer.save_yaml(flow, filepath)

def load_flow_yaml(filepath: str) -> Flow:
    """Load Flow from YAML file."""
    return _default_serializer.load_yaml(filepath)

def validate_yaml(yaml_str: str) -> bool:
    """Validate YAML string without loading."""
    return _default_serializer.validate_yaml(yaml_str)

def validate_yaml_file(filepath: str) -> bool:
    """Validate YAML file without loading."""
    return _default_serializer.validate_yaml_file(filepath)