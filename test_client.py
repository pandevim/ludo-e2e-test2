import os
from openai import OpenAI

# Initialize the OpenAI client pointing to the local tunnel
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="EMPTY"  # vLLM local API doesn't require a real API key
)

print("Sending request to Qwen 0.5B via HPC tunnel...")

try:
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a python function to calculate the Fibonacci sequence."}
        ],
        max_tokens=200,
    )
    
    print("\nStatus: Success!")
    print("-" * 50)
    print("Response from model:")
    print(response.choices[0].message.content.strip())
    print("-" * 50)
except Exception as e:
    print(f"\nStatus: Failed!")
    print(f"Error: {e}")
