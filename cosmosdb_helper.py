from azure.cosmos import CosmosClient, exceptions, PartitionKey
import json
import openai_helper
from dataclasses import dataclass, field, asdict, is_dataclass, fields
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path 
import os

env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)


endpoint = os.environ["AZURE_COSMOSDB_ENDPOINT"]
key = os.environ["AZURE_COSMOSDB_KEY"]
client = CosmosClient(endpoint, key)
database_name = 'vectordb'
database = client.get_database_client(database_name)
container_name_2partition = 'vectortest_hybridsearch_2partitions'
container_2partition = database.get_container_client(container_name_2partition)


print("CosmosDB endpoint: ", endpoint)

print("database_name: ", database_name)
print("container_name: ", container_name_2partition)

@dataclass
class Element:
    """Base class for all elements."""
    id: Optional[str] = None

@dataclass
class Xref(Element):
    id: Optional[str]
    xreflabel: str = ''
    linkend: str = ''

def get_cosmosdb_context(search_query, num_items=5):
    
    search_query2_embedded = openai_helper.generate_embeddings([search_query], 1536)
    search_query2_arr = search_query.split(" ")
    query_string = f"""
    SELECT TOP {num_items} c.section_id, c.section_title, c.para, c.xrefs
    FROM c
    ORDER BY RANK RRF(VectorDistance(c.paraVector, {search_query2_embedded[0].embedding}), 
    VectorDistance(c.summaryVector, {search_query2_embedded[0].embedding}), 
    VectorDistance(c.sectionVector, {search_query2_embedded[0].embedding}),
    VectorDistance(c.topicVector, {search_query2_embedded[0].embedding}),
    VectorDistance(c.keywordVector, {search_query2_embedded[0].embedding}),
    FullTextScore(c.para, {search_query2_arr}))
    """

    #print("Query: ", query_string)
    items = container_2partition.query_items( 
    query=query_string, 
    parameters=[], 
    enable_cross_partition_query=True)

    

    context = ""

    for item in items:
        
        context += f"""Main Section:
        Section Title: {item['section_title']}\n
        Section ID: {item['section_id']}\n
        Paragraph: {item['para']}\n
        Reference Sections: \n
        """

        doc = json.loads(json.dumps(item))
        
        if 'xrefs' in doc:
            xrefs = doc['xrefs']
            xrefs_list = json.loads(xrefs)
            xrefs = [Xref(**xref) for xref in xrefs_list]
            doc['xrefs'] = xrefs
            if len(doc['xrefs']) > 0:
                
                linkend_query_string = ""
                for xref in doc['xrefs']:
                    linkend_query_string += f"'{(xref.linkend)}',"
                
                linkend_query_string = f"SELECT c.section_title, c.section_id, c.para from c WHERE c.section_id IN ({linkend_query_string.rstrip(',')})"
                linked_items = container_2partition.query_items( 
                    query=linkend_query_string, 
                    parameters=[], 
                    enable_cross_partition_query=True)
                for linked_item in linked_items:
                    #print(f"Main Context: {context}")
                    context += f"""Section Title: {linked_item['section_title']}\n
                    Section ID: {linked_item['section_id']}\n
                    Paragraph: {linked_item['para']}\n
                    """
    return context