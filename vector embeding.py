from datasets import load_dataset

data = load_dataset("wikipedia", "20220301.simple", split='train[:5000]')

print("Data description:")
print(data)
print ("\n")

# function to calculate length of a chunk in tokens

import tiktoken 

tokenizer = tiktoken.get_encoding('p50k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


# function to split text into smaller chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

# Creating embeddings

from langchain.embeddings.openai import OpenAIEmbeddings

embed = OpenAIEmbeddings()

# Pinecone setup

import pinecone

index_name = 'langchain-test-index'

pinecone.init(
        api_key="8959372c-8528-4146-a414-db54df3ae152",  # find api key in console at app.pinecone.io
        environment="northamerica-northeast1-gcp"  # find next to api key in console
)

# create a new index
# commented as only need to run once

# pinecone.create_index(
#         name=index_name,
#         metric='dotproduct',
#         dimension=1536 # 1536 is the dimension of text-embedding-ada-002
# )

# connect to index
index = pinecone.Index(index_name)

print("Index Stats:")
print(index.describe_index_stats())
print ("\n")

# indexing process
# batched to speed up
# iterating through the data to add to our knowledge base, creating IDs, embeddings, and metadata
from tqdm.auto import tqdm
from uuid import uuid4

# commented as only need to run once

# number of chunks to batch together
batch_limit = 100

# contains chunks to be vectorised
texts = []
# holds meta data for each chunk
metadatas = []

for i, record in enumerate(tqdm(data)):
    # get metadata fields for this record
    metadata = {
        'wiki-id': str(record['id']),
        'source': record['url'],
        'title': record['title']
    }
    # create chunks
    record_texts = text_splitter.split_text(record['text'])
    # create individual metadata dicts for each chunk
    record_metadatas = [{
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(record_texts)]
    # append these to current batches
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)
    # if we have reached the batch_limit we can add texts
    if len(texts) >= batch_limit:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []

print(index.describe_index_stats())