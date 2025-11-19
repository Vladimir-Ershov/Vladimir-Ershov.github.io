"""
Test client for the Toy VLA API.
"""
import requests
import json
import sys
from typing import Optional


class VLAClient:
    """Client for interacting with the Toy VLA API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def get_status(self):
        """Get system status."""
        response = requests.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()

    def execute_task(
        self,
        command: str,
        context: Optional[str] = None,
        use_structured_output: bool = True
    ):
        """Execute a VLA task."""
        payload = {
            "command": command,
            "context": context,
            "use_structured_output": use_structured_output
        }

        response = requests.post(
            f"{self.base_url}/execute",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def clear_buffer(self):
        """Clear the image buffer."""
        response = requests.post(f"{self.base_url}/clear_buffer")
        response.raise_for_status()
        return response.json()


def print_result(result: dict):
    """Pretty print the result."""
    print("\n" + "="*80)
    print("TASK EXECUTION RESULT")
    print("="*80)

    print(f"\nTask ID: {result['task_id']}")
    print(f"Command: {result['command']}")

    if result.get('context'):
        print(f"Context: {result['context']}")

    if result.get('image_path'):
        print(f"Image: {result['image_path']}")

    if result.get('error'):
        print(f"\nERROR: {result['error']}")
    elif result.get('output'):
        output = result['output']
        print(f"\n--- ACTION PREDICTION ---")
        print(f"Action:       {output['action']}")
        print(f"Target:       {output['target']}")
        print(f"Destination:  {output.get('destination', 'N/A')}")
        print(f"Completion:   {output['completion']}%")
        print(f"Skill:        {output['current_skill']}")
        print(f"Failure:      {output['failure']}")

        if output.get('reasoning'):
            print(f"\nReasoning: {output['reasoning']}")

    print("="*80 + "\n")


def main():
    """Main test function."""
    print("Toy VLA System - Test Client")
    print("-" * 80)

    client = VLAClient()

    # Check status
    try:
        print("\n1. Checking system status...")
        status = client.get_status()
        print(f"   Status: {status['status']}")
        print(f"   Buffer size: {status['buffer_size']}")
        print(f"   API key configured: {status['api_key_configured']}")

        if not status['api_key_configured']:
            print("\n   WARNING: API key not configured!")
            print("   Set GOOGLE_API_KEY in .env file to test with real Gemini API")
            sys.exit(1)

        if status['buffer_size'] == 0:
            print("\n   WARNING: No images in buffer!")
            print("   Add images to test_images/ folder and wait a few seconds")
            sys.exit(1)

    except requests.exceptions.ConnectionError:
        print("   ERROR: Cannot connect to server!")
        print("   Make sure the server is running (python main.py)")
        sys.exit(1)

    # Test commands
    test_commands = [
        {
            "command": "Pick up the red cube",
            "context": "The robot is standing near the table"
        },
        {
            "command": "Move the red cube next to the blue sphere",
            "context": None
        },
        {
            "command": "Count all the yellow objects",
            "context": None
        },
        {
            "command": "Push the green block to the left",
            "context": "The table has multiple objects"
        }
    ]

    print("\n2. Testing task execution...")
    for i, test in enumerate(test_commands, 1):
        print(f"\n   Test {i}/{len(test_commands)}: {test['command']}")

        try:
            result = client.execute_task(
                command=test['command'],
                context=test['context']
            )
            print_result(result)

            # Pause between tests
            if i < len(test_commands):
                input("   Press Enter to continue to next test...")

        except Exception as e:
            print(f"   ERROR: {e}")
            continue

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
