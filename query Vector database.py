from langchain.embeddings.openai import OpenAIEmbeddings

embed = OpenAIEmbeddings()

import pinecone

index_name = 'langchain-test-index'

pinecone.init(
        api_key="8959372c-8528-4146-a414-db54df3ae152",  # find api key in console at app.pinecone.io
        environment="northamerica-northeast1-gcp"  # find next to api key in console
)

index = pinecone.Index(index_name)

print("Index Stats:")
print(index.describe_index_stats())
print ("\n")

from langchain.vectorstores import Pinecone

text_field = "text"

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# similarity search to return chunks of text without LLM

query = "who was Benito Mussolini?"

print(vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
))

print("\n")

# Generative Question Answering
# we pass our question to the LLM but instruct it to base the answer on the information returned from our knowledge base.

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# completion llm
llm = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

print(qa.run(query))
print("\n")

# adding citations to the response

from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

print(qa_with_sources(query))
