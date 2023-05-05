from langchain import ConversationChain, OpenAI
from langchain.chains.conversation.memory import ConversationSummaryMemory

# first initialize the large language model
llm = OpenAI(
	temperature=0,
	model_name="text-davinci-003"
)

#  initiate conversation chain with ConversationSummaryMemory
conversation = ConversationChain(
	llm=llm,
	memory=ConversationSummaryMemory(llm=llm)
)

print(conversation.memory.prompt.template)

# Adding a count tokens function to see number of tokens used in each interaction

from langchain.callbacks import get_openai_callback

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

print(count_tokens(
    conversation, 
    "Good morning AI!"
))

print(count_tokens(
    conversation, 
    "My interest here is to explore the potential of integrating Large Language Models with external knowledge"
))

print(count_tokens(
    conversation, 
    "I just want to analyze the different possibilities. What can you think of?"
))

print(count_tokens(
    conversation, 
    "Which data source types could be used to give context to the model?"
))

print(count_tokens(
    conversation, 
    "Good morning AI!"
))

# print memory buffer

print(conversation.memory.buffer)

# reduces token usage by sumarizing conversation, but removes information