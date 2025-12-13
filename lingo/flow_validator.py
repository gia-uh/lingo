from typing import Any, Dict, List

class ValidationError(Exception):
    """Custom exception for YAML validation errors."""
    def __init__(self, message: str, errors: List[str] = None):
        self.message = message
        self.errors = errors or []
        super().__init__(f"{message}\n" + "\n".join(f"  - {e}" for e in self.errors))


class FlowValidator:
    """
    Validates YAML structure against the Flow schema.
    """
    
    # Valid node types
    VALID_NODE_TYPES = {
        "append", "prepend", "reply", "invoke", "noop",
        "create", "sequence", "decide", "choose", "route"
    }
    
    # Required fields for each node type
    NODE_REQUIREMENTS = {
        "append": {"message"},
        "prepend": {"message"},
        "reply": {"instructions"},
        "invoke": {"tools"},
        "create": {"model", "instructions"},
        "sequence": {"nodes"},
        "decide": {"on_true", "on_false", "instructions"},
        "choose": {"choices", "instructions"},
        "route": {"flows"},
        "noop": set()  # No requirements
    }
    
    def validate_yaml(self, yaml_data: Dict[str, Any]) -> List[str]:
        """
        Validate YAML data against Flow schema.
        
        Args:
            yaml_data: Parsed YAML dictionary
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # 1. Validate top-level structure
        errors.extend(self._validate_top_level(yaml_data))
        
        # 2. Validate nodes if present
        if "nodes" in yaml_data and not errors:
            errors.extend(self._validate_nodes(yaml_data["nodes"]))
        
        return errors
    
    def _validate_top_level(self, data: Dict[str, Any]) -> List[str]:
        """Validate top-level fields."""
        errors = []
        
        # Required fields
        if "name" not in data:
            errors.append("Missing required field: 'name'")
        
        if "nodes" not in data:
            errors.append("Missing required field: 'nodes'")
        
        # Field types
        if "name" in data and not isinstance(data["name"], str):
            errors.append("Field 'name' must be a string")
        
        if "description" in data and not isinstance(data.get("description"), str):
            errors.append("Field 'description' must be a string if present")
        
        if "nodes" in data and not isinstance(data["nodes"], list):
            errors.append("Field 'nodes' must be a list")
        
        return errors
    
    def _validate_nodes(self, nodes: List[Dict[str, Any]]) -> List[str]:
        """Validate list of nodes."""
        errors = []
        
        if not nodes:
            errors.append("'nodes' list cannot be empty")
            return errors
        
        for i, node in enumerate(nodes):
            node_errors = self._validate_single_node(node, i)
            errors.extend(node_errors)
        
        return errors
    
    def _validate_single_node(self, node: Dict[str, Any], index: int) -> List[str]:
        """Validate a single node."""
        errors = []
        
        # Check node type
        if "type" not in node:
            errors.append(f"Node {index}: Missing 'type' field")
            return errors  # Can't validate further without type
        
        node_type = node["type"]
        
        # Validate node type
        if node_type not in self.VALID_NODE_TYPES:
            valid_types = ", ".join(sorted(self.VALID_NODE_TYPES))
            errors.append(
                f"Node {index}: Invalid type '{node_type}'. "
                f"Must be one of: {valid_types}"
            )
        
        # Check required fields for this node type
        if node_type in self.NODE_REQUIREMENTS:
            required = self.NODE_REQUIREMENTS[node_type]
            for field in required:
                if field not in node:
                    errors.append(f"Node {index} ({node_type}): Missing required field '{field}'")
        
        # Node-specific validations
        if node_type == "append":
            if "message" in node:
                errors.extend(self._validate_message(node["message"], f"Node {index} (append)"))
        
        elif node_type == "prepend":
            if "message" in node:
                errors.extend(self._validate_message(node["message"], f"Node {index} (prepend)"))
        
        elif node_type == "reply":
            if "instructions" in node:
                errors.extend(self._validate_instructions(node["instructions"], f"Node {index} (reply)"))
        
        elif node_type == "invoke":
            if "tools" in node:
                errors.extend(self._validate_tools(node["tools"], f"Node {index} (invoke)"))
        
        elif node_type == "create":
            if "model" in node:
                errors.extend(self._validate_model(node["model"], f"Node {index} (create)"))
            if "instructions" in node:
                errors.extend(self._validate_instructions(node["instructions"], f"Node {index} (create)"))
        
        elif node_type == "sequence":
            if "nodes" in node:
                errors.extend(self._validate_nodes_in_sequence(node["nodes"], f"Node {index} (sequence)"))
        
        elif node_type == "decide":
            if "on_true" in node:
                errors.extend(self._validate_single_node(node["on_true"], f"{index}.on_true"))
            if "on_false" in node:
                errors.extend(self._validate_single_node(node["on_false"], f"{index}.on_false"))
            if "instructions" in node:
                errors.extend(self._validate_instructions(node["instructions"], f"Node {index} (decide)"))
        
        elif node_type == "choose":
            if "choices" in node:
                errors.extend(self._validate_choices(node["choices"], f"Node {index} (choose)"))
            if "instructions" in node:
                errors.extend(self._validate_instructions(node["instructions"], f"Node {index} (choose)"))
        
        elif node_type == "route":
            if "flows" in node:
                errors.extend(self._validate_flows(node["flows"], f"Node {index} (route)"))
        
        return errors
    
    def _validate_message(self, message: Dict[str, Any], context: str) -> List[str]:
        """Validate a message dictionary."""
        errors = []
        
        if not isinstance(message, dict):
            errors.append(f"{context}: 'message' must be a dictionary")
            return errors
        
        # Check required fields
        if "role" not in message:
            errors.append(f"{context}: Message missing 'role' field")
        
        if "content" not in message:
            errors.append(f"{context}: Message missing 'content' field")
        
        # Validate role
        if "role" in message:
            role = message["role"]
            valid_roles = {"system", "user", "assistant", "tool"}
            if role not in valid_roles:
                errors.append(f"{context}: Invalid role '{role}'. Must be one of: {', '.join(valid_roles)}")
        
        # Validate content structure
        if "content" in message and isinstance(message["content"], dict):
            content = message["content"]
            if "type" not in content:
                errors.append(f"{context}: Message content missing 'type' field")
            elif content["type"] not in {"string", "dict", "list", "pydantic_model"}:
                errors.append(f"{context}: Invalid content type '{content['type']}'")
        
        return errors
    
    def _validate_instructions(self, instructions: List[Any], context: str) -> List[str]:
        """Validate instructions list."""
        errors = []
        
        if not isinstance(instructions, list):
            errors.append(f"{context}: 'instructions' must be a list")
            return errors
        
        if not instructions:
            errors.append(f"{context}: 'instructions' list cannot be empty")
        
        for i, inst in enumerate(instructions):
            if isinstance(inst, dict):
                if "type" not in inst:
                    errors.append(f"{context}: Instruction {i} missing 'type' field")
                elif inst["type"] not in {"string", "message"}:
                    errors.append(f"{context}: Instruction {i} has invalid type '{inst['type']}'")
                elif inst["type"] == "message":
                    errors.extend(self._validate_message(inst.get("value", {}), f"{context} instruction {i}"))
            elif not isinstance(inst, str):
                errors.append(f"{context}: Instruction {i} must be string or dictionary")
        
        return errors
    
    def _validate_tools(self, tools: List[Any], context: str) -> List[str]:
        """Validate tools list."""
        errors = []
        
        if not isinstance(tools, list):
            errors.append(f"{context}: 'tools' must be a list")
            return errors
        
        if not tools:
            errors.append(f"{context}: 'tools' list cannot be empty")
        
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                errors.append(f"{context}: Tool {i} must be a dictionary")
                continue
            
            if "type" not in tool:
                errors.append(f"{context}: Tool {i} missing 'type' field")
                continue
            
            tool_type = tool["type"]
            valid_tool_types = {"registered", "delegate_tool", "custom_tool"}
            
            if tool_type not in valid_tool_types:
                errors.append(f"{context}: Tool {i} has invalid type '{tool_type}'")
                continue
            
            # Check required fields based on type
            if tool_type == "registered":
                if "name" not in tool:
                    errors.append(f"{context}: Tool {i} (registered): Missing 'name' field")
            
            elif tool_type == "delegate_tool":
                required = {"name", "description", "target_module", "target_name"}
                for field in required:
                    if field not in tool:
                        errors.append(f"{context}: Tool {i} (delegate_tool): Missing '{field}' field")
            
            elif tool_type == "custom_tool":
                required = {"name", "description", "class"}
                for field in required:
                    if field not in tool:
                        errors.append(f"{context}: Tool {i} (custom_tool): Missing '{field}' field")
        
        return errors
    
    def _validate_model(self, model: Dict[str, Any], context: str) -> List[str]:
        """Validate model dictionary."""
        errors = []
        
        if not isinstance(model, dict):
            errors.append(f"{context}: 'model' must be a dictionary")
            return errors
        
        required_fields = {"module", "name"}
        for field in required_fields:
            if field not in model:
                errors.append(f"{context}: Model missing '{field}' field")
        
        return errors
    
    def _validate_nodes_in_sequence(self, nodes: List[Dict[str, Any]], context: str) -> List[str]:
        """Validate nodes inside a sequence."""
        errors = []
        
        if not isinstance(nodes, list):
            errors.append(f"{context}: 'nodes' must be a list")
            return errors
        
        if not nodes:
            errors.append(f"{context}: Sequence 'nodes' list cannot be empty")
        
        # Validate each node in the sequence
        for i, node in enumerate(nodes):
            node_errors = self._validate_single_node(node, f"{context}.node[{i}]")
            errors.extend(node_errors)
        
        return errors
    
    def _validate_choices(self, choices: Dict[str, Any], context: str) -> List[str]:
        """Validate choices dictionary."""
        errors = []
        
        if not isinstance(choices, dict):
            errors.append(f"{context}: 'choices' must be a dictionary")
            return errors
        
        if not choices:
            errors.append(f"{context}: 'choices' dictionary cannot be empty")
        
        for key, value in choices.items():
            if not isinstance(key, str):
                errors.append(f"{context}: Choice key must be string, got {type(key).__name__}")
            
            if not isinstance(value, dict):
                errors.append(f"{context}: Choice '{key}' value must be a dictionary")
                continue
            
            # Validate the node for this choice
            node_errors = self._validate_single_node(value, f"{context}.choice['{key}']")
            errors.extend(node_errors)
        
        return errors
    
    def _validate_flows(self, flows: List[Dict[str, Any]], context: str) -> List[str]:
        """Validate flows in a route node."""
        errors = []
        
        if not isinstance(flows, list):
            errors.append(f"{context}: 'flows' must be a list")
            return errors
        
        if len(flows) < 2:
            errors.append(f"{context}: Route needs at least 2 flows, got {len(flows)}")
        
        for i, flow in enumerate(flows):
            if not isinstance(flow, dict):
                errors.append(f"{context}: Flow {i} must be a dictionary")
                continue
            
            # Validate the flow structure
            flow_errors = self._validate_top_level(flow)
            for error in flow_errors:
                errors.append(f"{context} flow {i}: {error}")
            
            # Validate nodes in the flow
            if "nodes" in flow:
                node_errors = self._validate_nodes(flow["nodes"])
                for error in node_errors:
                    errors.append(f"{context} flow {i}: {error}")
        
        return errors
