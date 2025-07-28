from unittest import result
from langchain_openai import OpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {
        "input": "LangchainはChatGPT・Large Langage Model(LLM)を実利用をより柔軟に簡易に行うためのツール軍です",
        "output": "Langchainは、ChatGPTや大規模言語モデル（LLM）を実際の利用において、より柔軟で簡単に活用するためのツール群です。"
    }
]

prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="入力: {input}\n出力: {output}",
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt,
    prefix="以下の句読点の抜けた入力に句読点を追加してください。追加してよい句読点は「、」「。」のみです。ほかの句読点は使用しないでください。",
    suffix="入力: {input_string}\n出力:",
    input_variables=["input_string"]
)

llm = OpenAI()
formatted_prompt = few_shot_prompt.format(
    input_string="私は様々な機能がモジュールとして提供されているLangchainを使って、ChatGPTをより効果的に活用したいと考えています"
)
result = llm.invoke(formatted_prompt)
print("formatted_prompt:", formatted_prompt)
print("result:", result)

