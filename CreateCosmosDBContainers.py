from azure.cosmos import CosmosClient, exceptions, PartitionKey
import json
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential

import os
# Define your Cosmos DB account information
endpoint = os.getenv("CAIG_COSMOSDB_NOSQL_URI")
keyCredential = AzureKeyCredential(os.getenv("CAIG_COSMOSDB_NOSQL_KEY"))
client = CosmosClient(endpoint, os.getenv("CAIG_COSMOSDB_NOSQL_KEY"))
database_name = os.getenv("CAIG_GRAPH_SOURCE_DB")
containers = [
    {
        "container_name": "libraries",
        "partition_key": "/pk",
        "vector_embedding_policy": {
            "type": "AzureCosmosDBVectorEmbeddingPolicy",
            "vectorEmbeddings": [
                {
                    "path": "/embedding",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": 1536
                }
            ]
        },
        "vector_indexing_policy" : {
    
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
                    "path": "/embedding/*"
                },
               
            ],
            "vectorIndexes": [
                {
                    "path": "/embedding",
                    "type": "diskANN"
                }
                
            ]
    }
    },
    {
        "container_name": "config",
        "partition_key": "/pk",        
    },
    {
        "container_name": "conversations",
        "partition_key": "/pk",        
    },
    {
        "container_name": "feedback",
        "partition_key": "/conversation_id",        
    }
]


database = client.create_database_if_not_exists(id=database_name)


for c in containers:
    # Create the container if it doesn't exist
    print(f"Creating container '{c['container_name']}'...")
    if "vector_embedding_policy" in c and "vector_indexing_policy" in c:
        container = database.create_container_if_not_exists(id=c["container_name"], partition_key=PartitionKey(path=c["partition_key"]), 
                        vector_embedding_policy=c["vector_embedding_policy"],
                        indexing_policy=c["vector_indexing_policy"],
                        offer_throughput=400)
    else:
        container = database.create_container_if_not_exists(id=c["container_name"], partition_key=PartitionKey(path=c["partition_key"]), offer_throughput=400)
    print(f"Container '{c['container_name']}' created successfully.")
        
