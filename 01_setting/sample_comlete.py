import os
import json
from openai import OpenAI

# OpenAI API >= 1.0.0 does not support completions API....
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
response = client.completions.create(
    model="gpt-3.5-turbo-instruct", # gpt3x model only
    prompt="今日は調子がよく、",
    stop="。",
    max_tokens=100,
    n=2,
    temperature=0.5,
)

print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))