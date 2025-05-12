# %%
from azure.cosmos import CosmosClient, exceptions, PartitionKey
import json
from azure.identity import DefaultAzureCredential

import openai_helper 
import os
import pandas as pd
import uuid
import tiktoken
import asyncio

# Define your Cosmos DB account information
endpoint = "https://anildwacosmoswestus.documents.azure.com:443/"


# Initialize the Cosmos client
client = CosmosClient(endpoint, credential=DefaultAzureCredential())

# %%
database_name = 'booksdb'
container_name = 'books'

client.create_database_if_not_exists(id=database_name)
# Connect to the database and container
database = client.get_database_client(database_name)

# %%
vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path":"/textVector",
            "dataType":"float32",
            "distanceFunction":"cosine",
            "dimensions":1536
        }
    ]
}


vector_indexing_policy = {
    
    "indexingMode": "consistent",
    "automatic": True,
    "includedPaths": [
        {
            "path": "/*"
        }
    ],
    "excludedPaths": [
        {
            "path": "/_etag/?"
        },
        {
            "path": "/textVector/*"
        }
        
    ],
    "vectorIndexes": [
        {
            "path": "/textVector",
            "type": "quantizedFlat"
        }
    ]
}

full_text_paths_policy = {
   "defaultLanguage": "en-US",
   "fullTextPaths": [
       {
           "path": "/fileName",
           "language": "en-US"
       },
       {
           "path": "/text",
           "language": "en-US"
       }
   ]
}




vector_indexing_policy_diskANN = {
    
    "indexingMode": "consistent",
    "automatic": True,
    "includedPaths": [
        {
            "path": "/*"
        }
    ],
    "excludedPaths": [
        {
            "path": "/_etag/?"
        },
        {
            "path": "/textVector/*"
        }
    ],
    "fullTextIndexes": [
        {
            "path": "/text"
        }
    ],
    "vectorIndexes": [
        {
            "path": "/textVector",
            "type": "diskANN"
        }
    ]
}

# %%
for db in client.list_databases():
    print(db)

# %%
#database.delete_container(container=container_name)


container = None
try:
    container = database.create_container(id=container_name, partition_key=PartitionKey(path="/id"), 
                          vector_embedding_policy=vector_embedding_policy,
                          indexing_policy=vector_indexing_policy_diskANN,
                          full_text_policy=full_text_paths_policy,
                          offer_throughput=10000) 
except exceptions.CosmosResourceExistsError:
    print(f"Container {container_name} already exists. Using existing container.")
    container = database.get_container_client(container_name)



# read csv files 

encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

MAX_TOKENS = 8192


async def start_processing():
  count = 0
  csv_files_path='q1_files/'
  total=len(os.listdir(csv_files_path))

  for file in os.listdir(csv_files_path):
      df = pd.read_csv(csv_files_path + file)
      text = df['text'].iloc[0]
      tokens = encoding.encode(text)
      token_length = len(tokens)

      if len(tokens) > MAX_TOKENS:
          tokens = tokens[:MAX_TOKENS]
          text = encoding.decode(tokens)

      embedding_result = await openai_helper.generate_embeddings(text)
      book_item = {
        "id": str(uuid.uuid4()),
        "fileName": file,
        "text": text,
        "textVector": list(embedding_result[0].embedding),
      }
      json.dumps(book_item)
    
      try:    
        res = container.upsert_item(book_item)
      except:
        print(file)
        print(len(text))
        break
      count += 1
      print(f"processing {count} of {total}")
    



if __name__ == "__main__":
	asyncio.run(start_processing())



