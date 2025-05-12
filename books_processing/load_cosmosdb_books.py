# %%
from azure.cosmos import CosmosClient, exceptions, PartitionKey
import json
from azure.identity import DefaultAzureCredential

import openai_helper 
import os
import pandas as pd
import uuid

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

# %%
container = database.create_container(id=container_name, partition_key=PartitionKey(path="/id"), 
                          vector_embedding_policy=vector_embedding_policy,
                          indexing_policy=vector_indexing_policy_diskANN,
                          full_text_policy=full_text_paths_policy,
                          offer_throughput=10000) 

# %%
# read csv files 



count = 0
total=len(os.listdir('q1_files'))
for file in os.listdir('q1_files'):
    df = pd.read_csv('q1_files/' + file)
    


    book_item = {
        "id": str(uuid.uuid4()),
        "fileName": file,
        "text": df['text'].iloc[0],
        "textVector": openai_helper.generate_embeddings(df['text'].iloc[0])[0].embedding,
    }
    
    
    res = container.upsert_item(book_item)
    count += 1
    print(f"processing {count} of {total}")
    

# %%



