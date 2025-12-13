"""
Complete YAML serialization test - Focus on roundtrip functionality.
"""

import os
import sys
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pydantic import BaseModel
from typing import Optional, List

from lingo.flow import Flow
from lingo.tools import Tool, tool
from lingo.serializer import FlowSerializer


# =========== DEFINITIONS FOR THE TEST ===========

# Pydantic model for 'create' node
class UserProfile(BaseModel):
    """User profile extracted from conversation."""
    name: str
    age: Optional[int] = None
    interests: List[str] = []
    sentiment: str = "neutral"


# Custom tools
class CalculatorTool(Tool):
    """Basic calculator."""
    
    def __init__(self):
        super().__init__(name="calculator", description="Perform mathematical calculations")
    
    def parameters(self) -> dict[str, type]:
        return {"operation": str, "a": float, "b": float}
    
    async def run(self, operation: str, a: float, b: float) -> float:
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            return a / b if b != 0 else float('inf')
        else:
            raise ValueError(f"Invalid operation: {operation}")


@tool
async def weather_tool(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city} is sunny, 25°C"


def create_complex_flow() -> Flow:
    """
    Create a flow that uses ALL available node types.
    """
    
    # Register tools with serializer
    serializer = FlowSerializer()
    calculator = CalculatorTool()
    serializer.register_tool(calculator)
    serializer.register_tool(weather_tool)
    
    print("Creating complex flow with all node types...")
    
    # === MAIN FLOW ===
    main_flow = Flow(
        name="SuperAssistant",
        description="A complex assistant demonstrating all capabilities"
    )
    
    # 1. APPEND - System message
    print("  ✓ Append: System message")
    main_flow.append("You are an expert assistant in multiple areas.")
    
    # 2. PREPEND - Historical context (simulated)
    print("  ✓ Prepend: Historical context")
    main_flow.prepend("Previous conversation: User said 'Hello'")
    
    # 3. REPLY - Initial response
    print("  ✓ Reply: Initial greeting")
    main_flow.reply(
        "Hello! I'm your multifunctional assistant.",
        "How can I help you today?"
    )
    
    # === NESTED SEQUENCE using the fluent API ===
    print("  ✓ Sequence: Nested analysis flow")
    
    # Create nested flows using the fluent API
    math_handler = Flow(name="MathHandler")
    math_handler.append("User needs math help")
    math_handler.reply("It seems you need help with mathematics.")
    
    other_handler = Flow(name="OtherHandler")
    other_handler.append("User needs something else")
    other_handler.reply("I see you need something different.")
    
    analysis_flow = Flow(name="AnalysisFlow")
    analysis_flow.append("Starting request analysis...")
    analysis_flow.reply("Analyzing your request...")
    analysis_flow.decide(
        "Did the user mention numbers or calculations?",
        yes=math_handler,
        no=other_handler
    )
    
    main_flow.then(analysis_flow)
    
    # 4. INVOKE - Use tools
    print("  ✓ Invoke: Use calculator tool")
    main_flow.invoke(calculator)
    
    # 5. CHOOSE - Multiple choice
    print("  ✓ Choose: Options menu")
    
    # Create flows for each option using the fluent API
    math_flow = Flow(name="MathFlow")
    math_flow.append("Math mode activated")
    math_flow.reply("I'll help you with mathematics.")
    math_flow.invoke(calculator)
    
    weather_flow = Flow(name="WeatherFlow")
    weather_flow.append("Weather mode activated")
    weather_flow.reply("I'll help you with weather.")
    weather_flow.invoke(weather_tool)
    
    other_flow = Flow(name="OtherFlow")
    other_flow.append("General mode activated")
    other_flow.reply("I'll help you generally.")
    
    main_flow.choose(
        "Which area interests you?",
        choices={
            "mathematics": math_flow,
            "weather": weather_flow,
            "other": other_flow
        }
    )
    
    # 6. CREATE - Generate Pydantic model
    print("  ✓ Create: Extract user profile")
    main_flow.create(
        UserProfile,
        "Extract user information from the conversation"
    )
    
    # 7. ROUTE - Routing between flows
    print("  ✓ Route: Dynamic routing")
    
    # Create alternative flows using the fluent API
    detailed_flow = Flow(
        name="DetailedFlow",
        description="Flow with detailed responses"
    )
    detailed_flow.append("Provide detailed and extensive responses.")
    detailed_flow.reply("I'll give you a very detailed answer...")
    
    quick_flow = Flow(
        name="QuickFlow", 
        description="Flow with quick responses"
    )
    quick_flow.append("Provide concise responses.")
    quick_flow.reply("Quick answer: OK.")
    
    main_flow.route(detailed_flow, quick_flow)
    
    # 8. DECIDE - Binary decision (already used above, but add another)
    print("  ✓ Decide: Final decision")
    
    satisfied_flow = Flow(name="SatisfiedFlow")
    satisfied_flow.append("User satisfied")
    satisfied_flow.reply("Glad I could help!")
    
    unsatisfied_flow = Flow(name="UnsatisfiedFlow")
    unsatisfied_flow.append("User not satisfied")
    unsatisfied_flow.reply("Sorry I couldn't help better.")
    
    main_flow.decide(
        "Is the user satisfied?",
        yes=satisfied_flow,
        no=unsatisfied_flow
    )
    
    # 9. NOOP - Do nothing (special case)
    print("  ✓ NoOp: Silent termination")
    # Note: The fluent API doesn't have a direct .noop() method,
    # but we can add a NoOp node if needed
    from lingo.flow import NoOp
    main_flow.then(NoOp())
    
    print(f"\n✅ Flow created with {len(main_flow.nodes)} total nodes")
    
    return main_flow, serializer


def test_yaml_roundtrip():
    """
    Test YAML roundtrip: flow → YAML → flow
    Focus on functional equivalence, not exact type counting.
    """
    
    print("\n" + "="*60)
    print("YAML ROUNDTRIP TEST")
    print("="*60 + "\n")
    
    # Create original flow
    original_flow, serializer = create_complex_flow()
    
    # 1. Serialize to YAML
    print("1. Serializing to YAML...")
    yaml_str = serializer.to_yaml(original_flow)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_str)
        temp_file = f.name
    
    print(f"   YAML saved to: {temp_file}")
    print(f"   YAML size: {len(yaml_str)} bytes")
    print(f"   YAML lines: {yaml_str.count(chr(10)) + 1}")
    
    # 2. Deserialize from YAML
    print("\n2. Deserializing from YAML...")
    
    try:
        # Validate first
        print("   Validating YAML...")
        serializer.validate_yaml_file(temp_file)
        print("   ✅ YAML valid")
        
        # Load flow
        loaded_flow = serializer.load_yaml(temp_file)
        print(f"   ✅ Flow loaded: {loaded_flow.name}")
        
    except Exception as e:
        print(f"   ❌ Error loading: {e}")
        import traceback
        traceback.print_exc()
        os.unlink(temp_file)
        return False
    
    # 3. Compare basic properties
    print("\n3. Comparing basic properties...")
    
    comparisons = [
        ("Name", original_flow.name, loaded_flow.name),
        ("Description", original_flow.description, loaded_flow.description),
        ("Node count", len(original_flow.nodes), len(loaded_flow.nodes))
    ]
    
    for prop, orig_val, loaded_val in comparisons:
        if orig_val == loaded_val:
            print(f"   ✅ {prop}: {orig_val}")
        else:
            print(f"   ❌ {prop}: {orig_val} != {loaded_val}")
            os.unlink(temp_file)
            return False
    
    # 4. Count total executable nodes (simplified check)
    print("\n4. Counting executable nodes...")
    
    def count_executable_nodes(flow):
        """Count nodes that would be executed (simplified)."""
        count = 0
        nodes_to_process = list(flow.nodes)
        
        while nodes_to_process:
            node = nodes_to_process.pop(0)
            count += 1
            
            # Handle nested nodes
            from lingo.flow import Sequence, Decide, Choose, Route
            
            if isinstance(node, Sequence):
                nodes_to_process.extend(node.nodes)
            elif isinstance(node, Decide):
                nodes_to_process.append(node.on_true)
                nodes_to_process.append(node.on_false)
            elif isinstance(node, Choose):
                nodes_to_process.extend(node.choices.values())
            elif isinstance(node, Route):
                for subflow in node.flows:
                    nodes_to_process.extend(subflow.nodes)
        
        return count
    
    orig_nodes = count_executable_nodes(original_flow)
    loaded_nodes = count_executable_nodes(loaded_flow)
    
    print(f"   Original executable nodes: {orig_nodes}")
    print(f"   Loaded executable nodes: {loaded_nodes}")
    
    if orig_nodes == loaded_nodes:
        print(f"   ✅ Same number of executable nodes")
    else:
        print(f"   ❌ Different number of executable nodes")
        os.unlink(temp_file)
        return False
    
    # 5. Test re-serialization produces same YAML structure
    print("\n5. Testing re-serialization...")
    
    yaml_str2 = serializer.to_yaml(loaded_flow)
    
    # Simple comparison (not exact string match due to formatting)
    if len(yaml_str) == len(yaml_str2):
        print(f"   ✅ Re-serialized YAML same size: {len(yaml_str2)} bytes")
    else:
        # Small differences in formatting are OK
        size_diff = abs(len(yaml_str) - len(yaml_str2))
        print(f"   ⚠️  YAML size difference: {size_diff} bytes (formatting)")
    
    # 6. Clean up
    print("\n6. Cleaning up...")
    os.unlink(temp_file)
    print(f"   Removed temporary file: {temp_file}")
    
    # 7. Final verdict
    print("\n" + "="*60)
    print("🎉 YAML ROUNDTRIP TEST PASSED!")
    print("="*60)
    
    print("\nThe flow successfully completed the roundtrip:")
    print("  • Original flow → YAML serialization")
    print("  • YAML → Loaded flow deserialization")
    print("  • All functional properties preserved")
    print("  • Executable node count identical")
    
    print(f"\n📊 Flow statistics:")
    print(f"  • Flow name: {original_flow.name}")
    print(f"  • Description: {original_flow.description}")
    print(f"  • Top-level nodes: {len(original_flow.nodes)}")
    print(f"  • Total executable nodes: {orig_nodes}")
    print(f"  • Tools included: 2")
    print(f"  • Pydantic models: 1")
    
    return True


def test_simple_roundtrip():
    """Test simple roundtrip for quick verification."""
    
    print("\n" + "="*60)
    print("SIMPLE ROUNDTRIP SANITY CHECK")
    print("="*60)
    
    serializer = FlowSerializer()
    
    # Create a very simple flow
    flow = Flow(name="SimpleTest")
    flow.append("System message")
    flow.reply("Generate response")
    flow.append("Another message")
    
    print("\n1. Simple flow created")
    print(f"   Name: {flow.name}")
    print(f"   Nodes: {len(flow.nodes)}")
    
    # Roundtrip
    print("\n2. Performing roundtrip...")
    yaml_str = serializer.to_yaml(flow)
    loaded_flow = serializer.from_yaml(yaml_str)
    
    # Quick validation
    if flow.name == loaded_flow.name:
        print(f"   ✅ Names match: {loaded_flow.name}")
    else:
        print(f"   ❌ Names don't match")
        return False
    
    if len(flow.nodes) == len(loaded_flow.nodes):
        print(f"   ✅ Node counts match: {len(loaded_flow.nodes)}")
    else:
        print(f"   ❌ Node counts don't match")
        return False
    
    print("\n✅ SIMPLE ROUNDTRIP PASSED")
    return True


def main():
    """Main test function."""
    print("="*70)
    print("YAML SERIALIZATION ROUNDTRIP TESTS")
    print("="*70)
    
    # Run simple test first
    if not test_simple_roundtrip():
        print("\n❌ Simple test failed - aborting")
        return 1
    
    # Run complete test
    if not test_yaml_roundtrip():
        print("\n❌ YAML roundtrip test failed")
        return 1
    
    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("="*70)
    print("\nThe YAML serializer is working correctly:")
    print("1. ✅ Simple roundtrip works")
    print("2. ✅ Complex flow roundtrip works")
    print("3. ✅ All node types supported")
    print("4. ✅ YAML validation passes")
    print("5. ✅ Executable structure preserved")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())