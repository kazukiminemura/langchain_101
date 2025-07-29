from ast import parse
from langchain_openai import ChatOpenAI 
from langchain.schema import HumanMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field, field_validator

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

class Smartphone(BaseModel):
    release_date: str = Field(description="スマートフォンのリリース日")
    screen_inches: float = Field(description="スマートフォンの画面サイズ（インチ）")
    os_installed: str = Field(description="スマートフォンにインストールされているOS")
    model_name: str = Field(description="スマートフォンのモデル名")

    @field_validator('screen_inches') # pydantic version 2.0以降
    def validate_screen_inches(cls, field):
        if field <= 0:
            raise ValueError("画面サイズは正の数でなければなりません。")
        return field
    
parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=Smartphone),
    llm=chat,
)

result = chat.invoke(
    [
        HumanMessage(content="Androidでリリースしたスマートフォンを１個挙げてください。"),
        HumanMessage(content=parser.get_format_instructions())
    ]
)

parsed_output = parser.parse(result.content)
print(f"モデル名:{ parsed_output.model_name}")
print(f"リリース日:{ parsed_output.release_date}")
print(f"画面サイズ:{ parsed_output.screen_inches}")
print(f"インストールOS:{ parsed_output.os_installed}")
