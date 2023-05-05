# summarizes the earliest interactions in a conversation while maintaining the max_token_limit most recent tokens in the conversation.

from langchain import ConversationChain, OpenAI
from langchain.memory import ConversationSummaryBufferMemory
# first initialize the large language model
llm = OpenAI(
	temperature=0,
	model_name="text-davinci-003"
)

#  initiate conversation chain with ConversationBufferWindowMemory
# k=1 — this means the window will remember the single latest interaction between the human and AI.

conversation = ConversationChain(
	llm=llm,
	memory=ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=50
    ),
    verbose=True
)

conversation.predict(input="Good morning AI!")

conversation.predict(input="My interest here is to explore the potential of integrating Large Language Models with external knowledge")

conversation.predict(input="I just want to analyze the different possibilities. What can you think of?")

conversation.predict(input="Which data source types could be used to give context to the model?")

conversation.predict(input="What is my aim again?")