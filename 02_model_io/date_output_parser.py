from langchain_openai import ChatOpenAI 
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers import DatetimeOutputParser

output_parser = DatetimeOutputParser()

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = PromptTemplate.from_template(
    "{product}のリリース日を教えてください。"
)

result = chat.invoke(
    [
        HumanMessage(content=prompt.format(product="iPhone 15 Pro Max")),
        HumanMessage(content=output_parser.get_format_instructions()),
    ]
)

output = output_parser.parse(result.content)
print(output) # 2023-09-22T00:00:00+09:00