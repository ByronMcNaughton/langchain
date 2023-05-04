# Agents - decide the order to run queries
# https://python.langchain.com/en/latest/modules/agents/agents.html
# Tool: A function that performs a specific duty
# https://python.langchain.com/en/latest/modules/agents/tools/getting_started.html

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
# verbose -> prints thoughts and actions
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Now let's test it out!
agent.run("What was the high temperature in London yesterday in celsius? What is that number raised to the .023 power?")
