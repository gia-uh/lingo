"""
Test YAML flow execution with mocked LLM.
Validates that loaded flows can be executed with mock responses.
"""

import os
import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pydantic import BaseModel
from typing import List

from lingo.flow import Flow
from lingo.llm import LLM, Message
from lingo.engine import Engine
from lingo.tools import Tool, tool
from lingo.serializer import FlowSerializer


# =========== MOCK SETUP ===========

class MockLLM(LLM):
    """
    Mock LLM that returns predefined responses.
    Overrides chat and create methods for testing.
    """
    
    def __init__(self):
        self.model = "mock-model"
        self.chat_responses = []  # Queue of responses for chat()
        self.create_responses = {}  # Map of model types to responses
        self.next_decision = True  # Default decision
        self.next_choice = None    # Default choice
        self._chat_calls = []  # Track chat calls
        
    async def chat(self, messages: List[Message], **kwargs) -> Message:
        """Return next queued response or default."""
        self._chat_calls.append({
            'messages': messages,
            'kwargs': kwargs
        })
        
        if self.chat_responses:
            response = self.chat_responses.pop(0)
        else:
            response = "Mock response"
        
        return Message(role="assistant", content=response)
    
    async def create(self, model: type[BaseModel], messages: List[Message], **kwargs) -> BaseModel:
        """Return predefined model instance."""
        if model in self.create_responses:
            return self.create_responses[model]
        
        # Create default instance
        return model(name="Mock", value=42)


class MockTool(Tool):
    """Mock tool for testing."""
    
    def __init__(self, name="mock_tool", result="Mock result"):
        super().__init__(name=name, description=f"Mock tool: {name}")
        self.result = result
        self.last_called_with = None
    
    def parameters(self) -> dict[str, type]:
        return {"input": str}
    
    async def run(self, input: str) -> str:
        self.last_called_with = input
        return self.result


# =========== TEST FLOWS ===========

SIMPLE_FLOW_YAML = """
name: "SimpleChatFlow"
description: "Simple chat flow for testing"
nodes:
  - type: "append"
    message:
      role: "system"
      content:
        type: "string"
        value: "You are a test assistant"
  
  - type: "reply"
    instructions:
      - type: "string"
        value: "Respond to the user"
"""

DECISION_FLOW_YAML = """
name: "DecisionFlow"
description: "Flow with decision logic"
nodes:
  - type: "append"
    message:
      role: "system"
      content:
        type: "string"
        value: "Decision test"
  
  - type: "decide"
    on_true:
      type: "sequence"
      nodes:
        - type: "append"
          message:
            role: "system"
            content:
              type: "string"
              value: "Yes path"
        - type: "reply"
          instructions:
            - type: "string"
              value: "You said yes"
    on_false:
      type: "sequence"
      nodes:
        - type: "append"
          message:
            role: "system"
            content:
              type: "string"
              value: "No path"
        - type: "reply"
          instructions:
            - type: "string"
              value: "You said no"
    instructions:
      - type: "string"
        value: "Should we proceed?"
"""

TOOL_FLOW_YAML = """
name: "ToolFlow"
description: "Flow with tool invocation"
nodes:
  - type: "append"
    message:
      role: "system"
      content:
        type: "string"
        value: "Tool test"
  
  - type: "invoke"
    tools:
      - type: "registered"
        name: "mock_tool"
"""

# =========== TEST FUNCTIONS ===========

async def test_simple_flow_execution():
    """Test executing a simple flow with mock LLM."""
    
    print("=" * 60)
    print("TEST: SIMPLE FLOW EXECUTION")
    print("=" * 60)
    
    # Create mock LLM with predefined response
    mock_llm = MockLLM()
    mock_llm.chat_responses = ["Hello, I'm the mock assistant!"]
    
    # Create engine with mock LLM
    engine = Engine(mock_llm)
    
    # Load flow from YAML
    serializer = FlowSerializer()
    flow = serializer.from_yaml(SIMPLE_FLOW_YAML)
    
    print(f"\n1. Flow loaded: {flow.name}")
    print(f"   Description: {flow.description}")
    print(f"   Nodes: {len(flow.nodes)}")
    
    # Execute flow
    print("\n2. Executing flow...")
    
    initial_messages = [Message.user("Hello, test!")]
    context = await flow(engine, initial_messages)
    
    print(f"   Execution completed")
    print(f"   Final context messages: {len(context.messages)}")
    
    # Show all messages for debugging
    print(f"\n3. All messages in context:")
    for i, msg in enumerate(context.messages):
        content_preview = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
        print(f"   [{i}] {msg.role}: {content_preview}")
    
    # Verify results
    # 1. user (initial)
    # 2. system (append)
    # 3. assistant (reply)
    expected_min_messages = 3
    
    if len(context.messages) >= expected_min_messages:
        print(f"\n   ✅ Expected at least {expected_min_messages} messages")
        
        # Check message types
        roles = [msg.role for msg in context.messages]
        print(f"   Message roles: {roles}")
        
        # Verify we have all expected roles
        expected_roles = ['user', 'system', 'assistant']
        found_roles = [r for r in expected_roles if r in roles]
        
        if len(found_roles) == len(expected_roles):
            print(f"   ✅ All expected roles present: {expected_roles}")
        else:
            missing = set(expected_roles) - set(found_roles)
            print(f"   ❌ Missing roles: {missing}")
            return False
        
        # Check that we got the mock response
        last_message = context.messages[-1]
        if last_message.role == "assistant":
            print(f"   ✅ Last message is assistant response")
            
            # Check content matches our mock
            if "Hello, I'm the mock assistant!" in str(last_message.content):
                print(f"   ✅ Correct mock response received")
            else:
                print(f"   ❌ Response mismatch:")
                print(f"      Expected: 'Hello, I'm the mock assistant!'")
                print(f"      Got: '{str(last_message.content)[:50]}...'")
                return False
        else:
            print(f"   ❌ Last message not from assistant: {last_message.role}")
            return False
    else:
        print(f"   ❌ Not enough messages: {len(context.messages)}")
        return False
    
    print("\n✅ SIMPLE FLOW EXECUTION TEST PASSED")
    return True


async def test_decision_flow_execution():
    """Test executing a flow with decision logic."""
    
    print("\n" + "=" * 60)
    print("TEST: DECISION FLOW EXECUTION")
    print("=" * 60)
    
    # We need to mock the engine's decide method
    # since it uses the LLM internally
    
    with patch('lingo.engine.Engine.decide') as mock_decide:
        # Setup mock to return True
        mock_decide.return_value = True
        
        # Create LLM and engine
        llm = MockLLM()
        llm.chat_responses = ["Yes response"]
        engine = Engine(llm)
        
        # Load flow
        serializer = FlowSerializer()
        flow = serializer.from_yaml(DECISION_FLOW_YAML)
        
        print(f"\n1. Decision flow loaded: {flow.name}")
        
        # Execute
        print("\n2. Executing decision flow...")
        initial_messages = [Message.user("Test decision")]
        context = await flow(engine, initial_messages)
        
        print(f"   Execution completed")
        print(f"   Total messages: {len(context.messages)}")
        
        # Verify mock was called
        if mock_decide.called:
            print(f"   ✅ Decide method was called")
            
            # Check arguments
            args, kwargs = mock_decide.call_args
            context_arg = args[0] if args else kwargs.get('context')
            instructions = kwargs.get('instructions', ()) if kwargs else args[1:] if len(args) > 1 else ()
            
            if instructions and "Should we proceed?" in str(instructions[0]):
                print(f"   ✅ Correct decision prompt")
            else:
                print(f"   ❌ Wrong decision prompt: {instructions}")
                return False
        else:
            print(f"   ❌ Decide method not called")
            return False
        
        # Check that we have messages from the "yes" path
        system_messages = [m for m in context.messages if m.role == "system"]
        yes_path_message = any("Yes path" in str(m.content) for m in system_messages)
        
        if yes_path_message:
            print(f"   ✅ Yes path was executed")
        else:
            print(f"   ❌ Yes path not executed")
            return False
    
    print("\n✅ DECISION FLOW EXECUTION TEST PASSED")
    return True


async def test_tool_flow_execution():
    """Test executing a flow with tool invocation."""
    
    print("\n" + "=" * 60)
    print("TEST: TOOL FLOW EXECUTION")
    print("=" * 60)
    
    # Create mock tool
    mock_tool = MockTool(name="mock_tool", result="Tool executed successfully")
    
    # Create serializer and register tool
    serializer = FlowSerializer()
    serializer.register_tool(mock_tool)
    
    # Load flow
    flow = serializer.from_yaml(TOOL_FLOW_YAML)
    
    print(f"\n1. Tool flow loaded: {flow.name}")
    
    # We need to mock the engine's invoke method
    with patch('lingo.engine.Engine.invoke') as mock_invoke:
        # Setup mock to return successful tool result
        from lingo.tools import ToolResult
        mock_result = ToolResult(tool="mock_tool", result="Mock tool result")
        mock_invoke.return_value = mock_result
        
        # Create LLM and engine
        llm = MockLLM()
        engine = Engine(llm)
        
        # Execute
        print("\n2. Executing tool flow...")
        initial_messages = [Message.user("Use tool please")]
        context = await flow(engine, initial_messages)
        
        print(f"   Execution completed")
        print(f"   Total messages: {len(context.messages)}")
        
        # Verify invoke was called
        if mock_invoke.called:
            print(f"   ✅ Invoke method was called")
            
            # Check it was called with our tool
            args, kwargs = mock_invoke.call_args
            tool_arg = args[1] if len(args) > 1 else kwargs.get('tool')
            
            if tool_arg and tool_arg.name == "mock_tool":
                print(f"   ✅ Correct tool was invoked: {tool_arg.name}")
            else:
                print(f"   ❌ Wrong tool invoked")
                return False
        else:
            print(f"   ❌ Invoke method not called")
            return False
    
    print("\n✅ TOOL FLOW EXECUTION TEST PASSED")
    return True


async def test_complete_mock_execution():
    """Complete test with all mock types."""
    
    print("\n" + "=" * 60)
    print("COMPLETE MOCK EXECUTION TEST")
    print("=" * 60)
    
    # Create a simple flow programmatically
    flow = Flow(name="CompleteMockTest")
    flow.append("System message for mock test")
    flow.reply("Generate a response")
    
    print(f"\n1. Created test flow: {flow.name}")
    
    # Create fully mocked LLM
    mock_llm = AsyncMock(spec=LLM)
    
    # Mock the chat method to return specific response
    mock_response = Message.assistant("Mocked response from LLM")
    mock_llm.chat.return_value = mock_response
    
    # Create engine with mocked LLM
    engine = Engine(mock_llm)
    
    # Execute
    print("\n2. Executing with mocked LLM...")
    initial_messages = [Message.user("Test message")]
    context = await flow(engine, initial_messages)
    
    print(f"   Execution completed")
    print(f"   Context messages: {len(context.messages)}")
    
    # Verify LLM was called
    if mock_llm.chat.called:
        print(f"   ✅ LLM.chat() was called")
        
        # Check arguments
        args, _ = mock_llm.chat.call_args
        messages_arg = args[0] if args else None
        
        if messages_arg and len(messages_arg) > 0:
            print(f"   ✅ LLM received {len(messages_arg)} messages")
            
            # Check last message contains our mock response
            last_msg = context.messages[-1]
            if last_msg.content == "Mocked response from LLM":
                print(f"   ✅ Correct mock response in context")
            else:
                print(f"   ❌ Wrong response: {last_msg.content}")
                return False
        else:
            print(f"   ❌ No messages sent to LLM")
            return False
    else:
        print(f"   ❌ LLM.chat() not called")
        return False
    
    print("\n✅ COMPLETE MOCK EXECUTION TEST PASSED")
    return True


async def run_all_tests():
    """Run all mock execution tests."""
    
    print("YAML FLOW MOCK EXECUTION TESTS")
    print("=" * 70)
    
    results = []
    
    # Run tests
    tests = [
        ("Simple Flow Execution", test_simple_flow_execution),
        ("Decision Flow Execution", test_decision_flow_execution),
        ("Tool Flow Execution", test_tool_flow_execution),
        ("Complete Mock Execution", test_complete_mock_execution),
    ]
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 40)
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("\n🎉 ALL MOCK EXECUTION TESTS PASSED!")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED")
        for test_name, result in results:
            if not result:
                print(f"  ❌ {test_name}")
        return False


def main():
    """Main entry point."""
    print("YAML Flow Mock Execution Test Suite")
    print("=" * 70)
    
    # Run async tests
    try:
        success = asyncio.run(run_all_tests())
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())