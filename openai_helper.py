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

env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_key = os.getenv("AZURE_OPENAI_KEY", "") if len(os.getenv("AZURE_OPENAI_KEY", "")) > 0 else None
azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")



azure_openai_endpoint1 = os.environ["AZURE_OPENAI_ENDPOINT1"]
azure_openai_key1 = os.getenv("AZURE_OPENAI_KEY1", "") if len(os.getenv("AZURE_OPENAI_KEY1", "")) > 0 else None
azure_openai_deployment_name1 = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME1", "gpt-4o")

azure_openai_endpoint2 = os.environ["AZURE_OPENAI_ENDPOINT2"]
azure_openai_key2 = os.getenv("AZURE_OPENAI_KEY2", "") if len(os.getenv("AZURE_OPENAI_KEY2", "")) > 0 else None
azure_openai_deployment_name2 = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME2", "gpt-4o")

client = AzureOpenAI(
   api_version="2024-02-15-preview",
   azure_endpoint=azure_openai_endpoint,
   api_key=azure_openai_key,
   #azure_ad_token_provider=token_provider,   
)

client1 = AzureOpenAI(
   api_version="2024-02-15-preview",
   azure_deployment=azure_openai_deployment_name1,
   azure_endpoint=azure_openai_endpoint1,
   api_key=azure_openai_key1,
   #azure_ad_token_provider=token_provider,   
)

client2 = AzureOpenAI(
   api_version="2024-02-15-preview",
   azure_deployment=azure_openai_deployment_name2,
   azure_endpoint=azure_openai_endpoint2,
   api_key=azure_openai_key2,
   #azure_ad_token_provider=token_provider,   
)

aml_index_data = []
system_message = """
You are an AI assitant who can extract title, topics and cateogries from a document.
You will be given a document and you need to extract the title, topics and categories from the document in json format.

Title: Extract the title of the document that captures the information in the document.
Topics: Extract the topics from the document that best describe the content.
Categories: Extract the categories from the document that best describe the content.
Do not write ```json and ``` in your response.

json format:
{
    "title": "Document Title"
    "topics": ["topic1", "topic2"],
    "categories": ["category1", "category2"]
}
"""


tokenizer = tiktoken.get_encoding('cl100k_base')
def get_token_count(text):
    return len(tokenizer.encode(text))



def getOpenAIRespWithRetry(userQuery, systemMessage=system_message, deployed_model=azure_openai_deployment_name, streaming=False):
    max_retries = 5  # Maximum number of retry attempts
    retry_delay = 30  # Initial delay in seconds between retries

    for attempt in range(max_retries):
        try:
            # Choose a random client
            random_client = random.choice([client1, client2])
            print(f"Using Azure OpenAI: {random_client.base_url}")

            # Call the API
            completion = random_client.chat.completions.create(
                model=deployed_model,
                messages=[
                    {"role": "system", "content": systemMessage},
                    {"role": "user", "content": userQuery}
                ],
                temperature=0,
                max_tokens=4000,
                stream=streaming
            )

            # Return completion or the stream based on the `streaming` flag
            if streaming:
                return completion
            else:
                return completion.choices[0].message.content

        except RateLimitError as e:
            print(f"Rate limit reached (429 error). Attempt {attempt + 1} of {max_retries}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)  # Wait before retrying
            retry_delay *= 2  # Exponential backoff

        except Exception as e:
            print(f"An error occurred: {e}")
            break  # Exit if there's an error other than rate limiting

    print("Failed to get response after retries.")
    return None  # Return None if all retry attempts fail

def getOpenAIResp(userQuery, systemMessage=system_message, deployed_model=azure_openai_deployment_name, streaming=False):
    
    #random_client = random.choice([client1, client2])
    
    
    print(f"Using Azure OpenAI: {client.base_url}")
    completion = client.chat.completions.create(
            model=deployed_model,
            messages=[
                {
                    "role": "system",
                    "content": systemMessage
                },
                {
                    "role": "user",
                    "content": userQuery
                }
            ],
            temperature=0,
            max_tokens=4000,
            stream=streaming)
    #print(completion.choices[0].message.content)
    print(completion.prompt_filter_results)
    
    if streaming:
        return completion
    else:
        #print(f"azure token usage:{json.dumps(completion.usage.__dict__)}")
        return completion.choices[0].message.content


async def generate_embeddings(text_list, model="text-embedding-ada-002"):
    try:
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