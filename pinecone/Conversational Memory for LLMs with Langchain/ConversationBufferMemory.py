from langchain import OpenAI
from langchain.chains import ConversationChain

# first initialize the large language model
llm = OpenAI(
	temperature=0,
	model_name="text-davinci-003"
)

# now initialize the conversation chain
conversation = ConversationChain(llm=llm)

# print prompt template
print(conversation.prompt.template)

# "If the AI does not know the answer to a question, it truthfully says it does not know."
# -> reduce "halucinations"

from langchain.chains.conversation.memory import ConversationBufferMemory

conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

# the raw input of the past conversation between the human and AI is passed — in its raw form — to the {history} parameter

print(conversation_buf("Good morning AI!"))

# Adding a count tokens function to see number of tokens used in each interaction

from langchain.callbacks import get_openai_callback

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

# Sending multiple consecutive messages to check that it remembers previous queries

print (count_tokens(
    conversation_buf, 
    "My interest here is to explore the potential of integrating Large Language Models with external knowledge"
))

print(count_tokens(
    conversation_buf,
    "I just want to analyze the different possibilities. What can you think of?"
))

print(count_tokens(
    conversation_buf, 
    "Which data source types could be used to give context to the model?"
))

print(count_tokens(
    conversation_buf, 
    "What is my aim again?"
))

# printing memory buffer to view saved conversation

print(conversation_buf.memory.buffer)

# Storing everything gives the LLM the maximum amount of information
# However gets expensive and will quickly hit max tokens (4096)