import tiktoken

file_path = r"C:\Users\arun0\Videos\research_agent\fix_context.txt"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Use cl100k_base (used by GPT-4, GPT-3.5-turbo, etc.)
encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(content)

print(f"File: {file_path}")
print(f"Characters: {len(content)}")
print(f"Tokens: {len(tokens)}")