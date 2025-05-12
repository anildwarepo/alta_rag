import uuid
from datetime import datetime
import os
import json
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import pandas as pd
import os, json
import tiktoken
from dotenv import load_dotenv
from pathlib import Path 
from openai import RateLimitError 
import time
import random

load_dotenv()

azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_key = os.getenv("AZURE_OPENAI_KEY", "") if len(os.getenv("AZURE_OPENAI_KEY", "")) > 0 else None
azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
credential = DefaultAzureCredential()

token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/")

client = AzureOpenAI(
   api_version="2024-02-15-preview",
   azure_endpoint=azure_openai_endpoint,
   #api_key=azure_openai_key,
   azure_ad_token_provider=token_provider,   
)




async def generate_embeddings(text_list, model="text-embedding-ada-002"):
    try:

        print(f"azure openai endpoint:{azure_openai_endpoint}")
        # Send a batch of texts to the embedding API
        if text_list is None or len(text_list) == 0:
            return []
        #print("embedding model:", model)
        embeddings = client.embeddings.create(input=text_list, model=model).data
        return embeddings
    
    except RateLimitError as e:
        print("Rate limit reached (429 error).")
        raise
    
    except Exception as e:
        print("Error calling OpenAI:" + str(client.base_url))
        print(e)
        raise



