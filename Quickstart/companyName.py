import os

# Connect to OpenAi and ask a simple question

from langchain.llms import OpenAI

# High temperature = more Random
llm = OpenAI(temperature=0.8)

text = "What would be a good company name for a company that makes ai?"
print(llm(text))

# Create a prompt template

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

print(prompt.format(product="colorful socks"))

# Multi Step Workflow - Chains

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("colorful socks"))