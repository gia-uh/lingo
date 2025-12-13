"""
Minimal YAML serialization test.
Tests basic to_yaml and from_yaml functionality.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lingo.flow import Flow
from lingo.serializer import flow_to_yaml, flow_from_yaml


def test_to_yaml():
    """Test converting a flow to YAML."""
    
    print("Testing flow_to_yaml()...")
    
    # Create simplest possible flow
    flow = Flow(name="SimpleFlow")
    flow.append("You are helpful.")
    flow.reply("Hello")
    
    # Convert to YAML
    yaml_str = flow_to_yaml(flow)
    
    print(f"✅ Flow converted to YAML")
    print(f"   Flow name: {flow.name}")
    print(f"   YAML length: {len(yaml_str)} chars")
    
    # Show first few lines
    lines = yaml_str.split('\n')[:5]
    print(f"   First lines: {' | '.join(lines)}")
    
    return yaml_str


def test_from_yaml(yaml_str):
    """Test converting YAML back to flow."""
    
    print("\nTesting flow_from_yaml()...")
    
    # Convert YAML back to flow
    flow = flow_from_yaml(yaml_str)
    
    print(f"✅ YAML converted back to flow")
    print(f"   Loaded flow name: {flow.name}")
    print(f"   Nodes in flow: {len(flow.nodes)}")
    
    # Verify it has the right structure
    if len(flow.nodes) == 2:
        print(f"   ✅ Has 2 nodes as expected")
    else:
        print(f"   ❌ Expected 2 nodes, got {len(flow.nodes)}")
    
    return flow


def test_load_example_file():
    """Test loading from an example YAML file."""
    
    print("\nTesting load from example_flow.yaml...")
    
    # Load from file using flow_from_yaml
    with open("examples/sample_flow.yaml", "r") as f:
        yaml_content = f.read()
    
    flow = flow_from_yaml(yaml_content)
    
    print(f"✅ Loaded from file: {flow.name}")
    print(f"   File nodes: {len(flow.nodes)}")
    
    return flow



def test_minimal_roundtrip():
    """Test minimal roundtrip: flow → yaml → flow."""
    
    print("\nTesting minimal roundtrip...")
    
    # Create flow
    original = Flow(name="TestRoundtrip")
    original.append("Test message")
    
    # To YAML
    yaml_str = flow_to_yaml(original)
    
    # From YAML
    restored = flow_from_yaml(yaml_str)
    
    # Compare
    if original.name == restored.name:
        print("✅ Names match")
    else:
        print("❌ Names don't match")
    
    if len(original.nodes) == len(restored.nodes):
        print("✅ Node counts match")
    else:
        print("❌ Node counts don't match")


def main():
    """Run minimal tests."""
    
    print("=" * 50)
    print("MINIMAL YAML SERIALIZATION TEST")
    print("=" * 50)
    
    # Test 1: to_yaml
    yaml_str = test_to_yaml()
    
    # Test 2: from_yaml
    test_from_yaml(yaml_str)
    
    # Test 3: Load from example file
    test_load_example_file()
    
    # Test 4: Roundtrip
    test_minimal_roundtrip()
    
    print("\n" + "=" * 50)
    print("✅ ALL MINIMAL TESTS COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()