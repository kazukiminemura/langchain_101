from langchain import PromptTemplate

prompt = PromptTemplate(
    template="{product}はどこの会社が開発した製品ですか？",
    input_variables=[
        "product"
        ]
)

print(prompt.format(product="iPhone"))  # Outputs: iPhoneはどこの会社が開発した製品ですか？
print(prompt.format(product="Xperia"))  # Outputs: Xperiaはどこの会社が開発した製品ですか？