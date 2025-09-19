from ollama import Client

client = Client()

response = client.chat(
     model="gemma3:12b", 
     messages=[{"role": "user", "content": "What is capital of Bangladesh? Ans in very short."}]
)

print(response.message.content)
