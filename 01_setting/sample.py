import json
import openai

# OpenAI API >= 1.0.0
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "tell me about iPhone 8 release date",
        },
    ],
    max_tokens=100,
    temperature=1,
    n=2,
)

print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))