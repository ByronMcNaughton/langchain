from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

llm = OpenAI(temperature=0.9)

# create template for prompt
prompt = PromptTemplate(
    input_variables=["name", "occupation"],
    template="Find me all information on {name}, their ocupation is {occupation}. make me notes on them including thier linkedin",
)

# Load tools
tools = load_tools(["serpapi"], llm=llm)

# Initalize agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

print(agent.run(prompt.format(name="Jared forrest", occupation="software engineer")))