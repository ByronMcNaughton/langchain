from datasets import load_dataset

data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]')

print(data)
print(data[6])

# Data Chunking

# function to calculate number to tokens

import tiktoken 

tokenizer = tiktoken.get_encoding('p50k_base')


# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

print(tiktoken_len("hello I am a chunk of text and using the tiktoken_len function "
             "we can find the length of this chunk of text in tokens"))


# create RecursiveCharacterTextSplitter

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

# Split chunks

chunks = text_splitter.split_text(data[6]['text'])[:3]
print(chunks)

# verifying chunk sizes
for chunk in chunks:
    print(tiktoken_len(chunk))


# Creating embeddings

from langchain.embeddings.openai import OpenAIEmbeddings

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings()

texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here'
]

res = embed.embed_documents(texts)
print(len(res))
print(len(res[0]))

# Pinecone vector database

import pinecone

index_name = 'langchain-retrieval-augmentation'

pinecone.init(
        api_key="",  # find api key in console at app.pinecone.io
        environment="northamerica-northeast1-gcp"  # find next to api key in console
)

# create a new index
# commented as only need to run once

# pinecone.create_index(
#         name=index_name,
#         metric='dotproduct',
#         dimension=len(res[0]) # 1536 dim of text-embedding-ada-002
# )

# connect to new index
index = pinecone.Index(index_name)

print(index.describe_index_stats())

# indexing process
# batched to speed up
# iterating through the data to add to our knowledge base, creating IDs, embeddings, and metadata
from tqdm.auto import tqdm
from uuid import uuid4

# commented as only need to run once

# batch_limit = 100

# texts = []
# metadatas = []

# for i, record in enumerate(tqdm(data)):
#     # first get metadata fields for this record
#     metadata = {
#         'wiki-id': str(record['id']),
#         'source': record['url'],
#         'title': record['title']
#     }
#     # now we create chunks from the record text
#     record_texts = text_splitter.split_text(record['text'])
#     # create individual metadata dicts for each chunk
#     record_metadatas = [{
#         "chunk": j, "text": text, **metadata
#     } for j, text in enumerate(record_texts)]
#     # append these to current batches
#     texts.extend(record_texts)
#     metadatas.extend(record_metadatas)
#     # if we have reached the batch_limit we can add texts
#     if len(texts) >= batch_limit:
#         ids = [str(uuid4()) for _ in range(len(texts))]
#         embeds = embed.embed_documents(texts)
#         index.upsert(vectors=zip(ids, embeds, metadatas))
#         texts = []
#         metadatas = []
# 
# print(index.describe_index_stats())

from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# similarity search to return chunks of text without LLM

query = "who was Benito Mussolini?"

vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)

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

# adding citations to the response

from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

print(qa_with_sources(query))
