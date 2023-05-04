# Short term memory using Conversation chain
# remembers all previous inputs and outputs, adding them to context

from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)

output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
print(output)

output = conversation.predict(input="Testing Langchain")
print(output)