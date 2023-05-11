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

print(tools[0].name, tools[0].description)

# Initalise zero-shot-react-description - no memory

from langchain.agents import initialize_agent

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)

print(zero_shot_agent("what is (4.5*2.1)^2.2?"))

print(zero_shot_agent("if Mary has four apples and Giorgio brings two and a half apple "
                "boxes (apple box contains eight apples), how many apples do we "
                "have?"))

# add a LLM tool

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# initialize the LLM tool
llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='use this tool for general purpose queries and logic'
)

tools.append(llm_tool)

# reinitialize the agent
zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)

print(zero_shot_agent("what is the capital of Norway?"))
print(zero_shot_agent("what is (4.5*2.1)^2.2?"))

