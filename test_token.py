# test_token.py
import os
from dotenv import load_dotenv

load_dotenv()

token = os.environ.get("OPENROUTER_API_KEY", "NOT FOUND")
print(f"Token found: {'YES' if token != 'NOT FOUND' else 'NO'}")
print(f"Token starts with: {token[:8]}...")
print(f"Token length: {len(token)}")