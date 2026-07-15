"""
Quick test: verify the LLM connection to GLM-5.2 via NVIDIA NIM.
"""
import sys
sys.path.insert(0, ".")

from app.config import settings
from app.llm import get_llm

print("=" * 60)
print("LLM Connection Test")
print("=" * 60)
print(f"Provider:      {settings.llm_provider}")
print(f"Chat Model:    {settings.nvidia_chat_model}")
print(f"Base URL:      {settings.nvidia_base_url}")
print(f"API Key:       {settings.nvidia_api_key[:20]}... (truncated)")
print("=" * 60)

try:
    llm = get_llm()
    print("\n✅ LLM instance created successfully.")

    # Simple test message
    messages = [
        {"role": "user", "content": "Say 'Hello, connection is working!' in exactly those words."}
    ]

    print("\n📤 Sending test message...")
    response = llm.generate(messages)
    print(f"\n📥 Response:\n{response}")

    print("\n" + "=" * 60)
    print("✅ CONNECTION SUCCESSFUL — GLM-5.2 is responding!")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ CONNECTION FAILED: {type(e).__name__}: {e}")
    print("=" * 60)
    sys.exit(1)