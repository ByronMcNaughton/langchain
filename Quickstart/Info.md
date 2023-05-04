CompanyName.py
A service that generates a company name based on what the company makes.
requires:
(VSCode)
$env:OPENAI_API_KEY = "KEY"
or alternative

Temperature.py
A service that looks up the temperature
requires:
(VSCode)
$env:SERPAPI_API_KEY = "KEY"
or alternative

conversation.py
A simple chatbot which remembers previous messages

messages.py
Different types of messages and responses

chatPromptTemplates.py
Creating prompts and storing as templates, then querying with this template
Combining multiple templates to create a prompt

ChatAgent.py
Agent combined with chat model, using tools to search google and do maths