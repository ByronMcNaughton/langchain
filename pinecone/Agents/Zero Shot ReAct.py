# Initalise LLM

from langchain import OpenAI

llm = OpenAI(
    temperature=0,
    model_name="text-davinci-003"
)

# initalising llm_math agent

from langchain.chains import LLMMathChain
from langchain.agents import Tool

llm_math = LLMMathChain(llm=llm)

# initialize the math tool

from langchain.agents import load_tools

tools = load_tools(
    ['llm-math'],
    llm=llm
)