# examples/test_validation.py
"""
Test YAML validation functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lingo.serializer import FlowSerializer, ValidationError, validate_yaml, validate_yaml_file


def test_validation():
    """Test various validation scenarios."""
    
    serializer = FlowSerializer()
    
    print("=== Testing YAML Validation ===\n")
    
    # Test 1: Valid YAML
    print("1. Testing valid YAML...")
    valid_yaml = """
name: "TestFlow"
description: "A test flow"
nodes:
  - type: "append"
    message:
      role: "system"
      content:
        type: "string"
        value: "Test message"
  - type: "reply"
    instructions:
      - type: "string"
        value: "Test instruction"
"""
    
    try:
        serializer.validate_yaml(valid_yaml)
        print("✅ Valid YAML passes validation\n")
    except ValidationError as e:
        print(f"❌ Unexpected error: {e}\n")
    
    # Test 2: Missing required field
    print("2. Testing missing 'name' field...")
    missing_name = """
description: "Missing name"
nodes: []
"""
    
    try:
        serializer.validate_yaml(missing_name)
        print("❌ Should have failed (missing name)")
    except ValidationError as e:
        print("✅ Correctly caught error:")
        for error in e.errors[:1]:
            print(f"  - {error}")
        print()
    
    # Test 3: Invalid node type
    print("3. Testing invalid node type...")
    invalid_type = """
name: "Test"
nodes:
  - type: "replay"  # Typo: should be "reply"
    instructions: ["Test"]
"""
    
    try:
        serializer.validate_yaml(invalid_type)
        print("❌ Should have failed (invalid type)")
    except ValidationError as e:
        print("✅ Correctly caught error:")
        for error in e.errors[:2]:  # Show first 2 errors
            print(f"  - {error}")
        print()
    
    # Test 4: Missing required field in node
    print("4. Testing missing field in 'decide' node...")
    missing_field = """
name: "Test"
nodes:
  - type: "decide"
    on_true:
      type: "noop"
    # Missing: on_false, instructions
"""
    
    try:
        serializer.validate_yaml(missing_field)
        print("❌ Should have failed (missing fields)")
    except ValidationError as e:
        print("✅ Correctly caught error:")
        for error in e.errors[:2]:
            print(f"  - {error}")
        print()
    
    # Test 5: Invalid YAML syntax
    print("5. Testing invalid YAML syntax...")
    invalid_syntax = """
name: "Test"
nodes: [
  type: "append"  # Invalid YAML list syntax
"""
    
    try:
        serializer.validate_yaml(invalid_syntax)
        print("❌ Should have failed (invalid syntax)")
    except ValidationError as e:
        print(f"✅ Correctly caught syntax error: {e.message.split(':')[0]}")
        print()
    
    # Test 6: Empty nodes list
    print("6. Testing empty nodes list...")
    empty_nodes = """
name: "Test"
nodes: []
"""
    
    try:
        serializer.validate_yaml(empty_nodes)
        print("❌ Should have failed (empty nodes)")
    except ValidationError as e:
        print("✅ Correctly caught error:")
        for error in e.errors[:1]:
            print(f"  - {error}")
        print()
    
    # Test 7: Nested validation in sequence
    print("7. Testing nested validation in sequence...")
    nested_error = """
name: "Test"
nodes:
  - type: "sequence"
    nodes:
      - type: "append"
        # Missing: message
"""
    
    try:
        serializer.validate_yaml(nested_error)
        print("❌ Should have failed (nested error)")
    except ValidationError as e:
        print("✅ Correctly caught nested error:")
        # Show error with context
        for error in e.errors:
            if "append" in error:
                print(f"  - {error}")
        print()
    
    # Test 8: Choice validation
    print("8. Testing choice validation...")
    invalid_choice = """
name: "Test"
nodes:
  - type: "choose"
    choices:
      option1: "not a dict"  # Should be dict
    instructions: ["Choose"]
"""
    
    try:
        serializer.validate_yaml(invalid_choice)
        print("❌ Should have failed (invalid choice)")
    except ValidationError as e:
        print("✅ Correctly caught error:")
        for error in e.errors[:1]:
            print(f"  - {error}")
        print()
    
    # Test 9: Route validation
    print("9. Testing route validation...")
    invalid_route = """
name: "Test"
nodes:
  - type: "route"
    flows:
      - name: "Only one flow"  # Need at least 2
"""
    
    try:
        serializer.validate_yaml(invalid_route)
        print("❌ Should have failed (need 2+ flows)")
    except ValidationError as e:
        print("✅ Correctly caught error:")
        for error in e.errors[:1]:
            print(f"  - {error}")
        print()
    
    print("=== All validation tests completed ===")


def test_validation_functions():
    """Test convenience validation functions."""
    
    print("\n=== Testing Convenience Functions ===\n")
    
    # Create a test file
    test_yaml = """
name: "ConvenienceTest"
nodes:
  - type: "append"
    message:
      role: "system"
      content:
        type: "string"
        value: "Test"
"""
    
    # Test validate_yaml function
    print("1. Testing validate_yaml()...")
    try:
        result = validate_yaml(test_yaml)
        print(f"✅ validate_yaml() returned: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with invalid YAML
    print("\n2. Testing validate_yaml() with error...")
    invalid = """
name: "Test"
nodes:
  - type: "invalid_type"
"""
    
    try:
        result = validate_yaml(invalid)
        print(f"❌ Should have raised error, got: {result}")
    except ValidationError as e:
        print(f"✅ Correctly raised ValidationError")
        print(f"  First error: {e.errors[0] if e.errors else 'No errors'}")
    
    # Test file validation
    print("\n3. Testing file validation...")
    test_file = "test_validation.yaml"
    
    try:
        # Write test file
        with open(test_file, 'w') as f:
            f.write(test_yaml)
        
        # Validate file
        result = validate_yaml_file(test_file)
        print(f"✅ validate_yaml_file() returned: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"  Cleaned up {test_file}")
    
    print("\n=== Convenience functions test completed ===")


def test_error_messages():
    """Test helpful error messages."""
    
    print("\n=== Testing Helpful Error Messages ===\n")
    
    serializer = FlowSerializer()
    
    # Common typo: "replay" instead of "reply"
    typo_yaml = """
name: "TypoTest"
nodes:
  - type: "replay"  # Common typo
    instructions: ["Test"]
"""
    
    print("1. Testing typo suggestion...")
    try:
        serializer.validate_yaml(typo_yaml)
    except ValidationError as e:
        print("✅ Got validation error with suggestion:")
        for error in e.errors:
            if "Did you mean" in error:
                print(f"  - {error}")
    
    # Missing field with clear message
    missing_yaml = """
name: "MissingTest"
nodes:
  - type: "decide"
    on_true:
      type: "noop"
    # Missing on_false and instructions
"""
    
    print("\n2. Testing missing field messages...")
    try:
        serializer.validate_yaml(missing_yaml)
    except ValidationError as e:
        print("✅ Clear error messages:")
        for error in e.errors[:2]:  # Show first 2
            print(f"  - {error}")
    
    # Invalid value
    invalid_value = """
name: "ValueTest"
nodes:
  - type: "append"
    message:
      role: "invalid_role"  # Invalid role
      content:
        type: "string"
        value: "Test"
"""
    
    print("\n3. Testing invalid value messages...")
    try:
        serializer.validate_yaml(invalid_value)
    except ValidationError as e:
        print("✅ Value validation messages:")
        for error in e.errors:
            if "role" in error:
                print(f"  - {error}")


if __name__ == "__main__":
    test_validation()
    test_validation_functions()
    test_error_messages()