# Parametric knowledge — the knowledge mentioned above is anything that has been learned by the model during training time and is stored within the model weights (or parameters).
# Source knowledge — any knowledge provided to the model at inference time via the input prompt.

# giving the ai examples

from langchain import OpenAI


prompt = """The following is a conversation with an AI assistant.
The assistant is typically sarcastic and witty, producing creative 
and funny responses to the users questions. Here are some examples: 

User: What is the meaning of life?
AI: """

# initialize the models
openai = OpenAI(
    model_name="text-davinci-003"
)

openai.temperature = 1.0  # increase creativity/randomness of output

print(openai(prompt))

# Giving the ai some more info - example responses to simular questions

prompt = """The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples: 

User: How are you?
AI: I can't complain but sometimes I still do.

User: What time is it?
AI: It's time to get a watch.

User: What is the meaning of life?
AI: """

print(openai(prompt))