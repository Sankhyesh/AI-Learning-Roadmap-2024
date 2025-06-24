from openai import OpenAI
import os

# Use host.docker.internal to connect to services on host machine
client = OpenAI(
    base_url="http://host.docker.internal:12434/engines/v1",
    api_key='docker' 
)

completion = client.chat.completions.create(
    model="ai/qwen3:0.6B-F16",
    messages=[
        {"role": "system", "content": "understand the data and provide insights"},
        {"role": "user", "content": ""}
    ]
)

print(completion.choices[0].message.content)
