from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex,
    AzureOpenAIVectorizer,
    AzureOpenAIParameters
)
import json
from dotenv import load_dotenv
from pathlib import Path 
import os
import requests
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.search.documents.models import VectorizableTextQuery

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
import os
import openai_helper
from datetime import datetime
import uuid

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

load_dotenv(override=True) # take environment variables from .env.

# The following variables from your .env file are used in this notebook
azure_search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY", "")) if len(os.getenv("AZURE_SEARCH_ADMIN_KEY", "")) > 0 else DefaultAzureCredential()
index_name = "aml_index_2" #os.getenv("AZURE_SEARCH_INDEX", "vectest")
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_key = os.getenv("AZURE_OPENAI_KEY", "") if len(os.getenv("AZURE_OPENAI_KEY", "")) > 0 else None
azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
azure_openai_embedding__large_deployment = os.getenv("AZURE_OPENAI_3_LARGE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
azure_openai_embedding__small_deployment = os.getenv("AZURE_OPENAI_3_LARGE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
azure_openai_embedding_large_dimensions = int(os.getenv("AZURE_OPENAI_EMBEDDING_LARGE_DIMENSIONS", 3072))
azure_openai_embedding_small_dimensions = int(os.getenv("AZURE_OPENAI_EMBEDDING_SMALLDIMENSIONS", 1536))
embedding_model_name = os.getenv("AZURE_OPENAI_3_LARGE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
azure_document_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://document-intelligence.api.cognitive.microsoft.com/")
azure_document_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")



print(f"Azure Search EndPoint: {azure_search_endpoint}")
print(f"Index Name: {index_name}")
print(f"Azure OpenAI Endpoint: {azure_openai_endpoint}")
print(f"Azure OpenAI Large Embedding Deployment: {azure_openai_embedding__large_deployment}")
print(f"Azure OpenAI Small Embedding Deployment: {azure_openai_embedding__small_deployment}")
print(f"Embedding Model Name: {embedding_model_name}")
print(f"Azure OpenAI API Version: {azure_openai_api_version}")
print(f"Azure Document Intelligence Endpoint: {azure_document_intelligence_endpoint}")

doc_intelli_credential = AzureKeyCredential(azure_document_intelligence_key)
document_intelligence_client = DocumentIntelligenceClient(azure_document_intelligence_endpoint, doc_intelli_credential)

index_client = SearchIndexClient(
    endpoint=azure_search_endpoint, credential=credential)


def create_simple_index(index_name: str, analyzer_name: str = "en.microsoft", language_suffix: str = "en"):
        index_schema = {
        "name": index_name,
        "fields": [
            {
                "name": "id",
                "type": "Edm.String",
                "key": True,
                "sortable": True,
                "filterable": True,
                "facetable": True
            },
            # Existing fields
            # Adding the new fields as searchable text fields
            {
                "name": "content",
                "type": "Edm.String",
                "searchable": True
            },
            
            # Vector fields for embeddings
            {
                "name": "contentVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": 1536,
                "vectorSearchProfile": "amlHnswProfile"
                
            }
            # Existing fields such as lastUpdated, suggesters, scoringProfiles, etc.
        ],
        "scoringProfiles": [
            
        ],
        "suggesters": [
            
        ],
        "vectorSearch": {
                "algorithms": [
                    {
                        "name": "amlHnsw",
                        "kind": "hnsw",
                        "hnswParameters": {
                        "m": 4,
                        "metric": "cosine"
                        }
                    }
                
                ],
                "profiles": [
                    {
                        "name": "amlHnswProfile",
                        "algorithm": "amlHnsw",
                        "vectorizer": "amlVectorizer"
                    }
                
                ], 
                "vectorizers": [
                    {
                        "name":"amlVectorizer",
                        "kind":"azureOpenAI",
                        "azureOpenAIParameters": {
                            "resourceUri": azure_openai_endpoint,
                            "deploymentId": azure_openai_embedding__large_deployment,
                            "modelName": embedding_model_name,
                            "apiKey": azure_openai_key
                        }
                    }
                ]
                
    },
        "semantic": {
            "configurations": [
                {
                    "name": "aml-semantic-config",
                    "prioritizedFields": {
                        "titleField": {
                            "fieldName": "content"
                        },
                        "prioritizedKeywordsFields": [
                            {
                                "fieldName": "content"
                            }
                           
                        ],
                        "prioritizedContentFields": [
                            {
                                "fieldName": f"content"
                            }
                        ]
                    }
                }
            ]
        }
    }



        headers = {'Content-Type': 'application/json',
                'api-key': os.getenv("AZURE_SEARCH_ADMIN_KEY", "") }
        # Create Index
        url = azure_search_endpoint + "/indexes/" + index_name + "?api-version=2024-07-01"


        response = requests.get(url, headers=headers)
        if response.status_code == 404:
            response  = requests.put(url, headers=headers, json=index_schema)
            index = response.json()
            print(index)
        else:
            print("Index already exists")



def create_index(index_name: str, analyzer_name: str = "en.microsoft", language_suffix: str = "en"):
        index_schema = {
        "name": index_name,
        "fields": [
            {
                "name": "id",
                "type": "Edm.String",
                "key": True,
                "sortable": True,
                "filterable": True,
                "facetable": True
            },
            # Existing fields
            # Adding the new fields as searchable text fields
            {
                "name": "part_title",
                "type": "Edm.String",
                "searchable": True
            },
            {
                "name": "chapter_title",
                "type": "Edm.String",
                "searchable": True
            },
            {
                "name": "section_title",
                "type": "Edm.String",
                "searchable": True
            },
            {
                "name": "keywords",
                "type": "Collection(Edm.String)",
                "searchable": True
            },
            {
                "name": "para",
                "type": "Edm.String",
                "searchable": True
            },
            {
                "name": "topics",
                "type": "Collection(Edm.String)",
                "searchable": True
            },
            {
                "name": "summary",
                "type": "Edm.String",
                "searchable": True
            },
            {
                "name": "lastUpdated",
                "type": "Edm.DateTimeOffset"
            
            },
            {
                "name": "category",
                "type": "Collection(Edm.String)",
                "filterable": True,
                "searchable": True
            },
            {
                "name": "part_id",
                "type": "Edm.String",
                "searchable": True
            },
            {
                "name": "chapter_id",
                "type": "Edm.String",
                "searchable": True
            },
            {
                "name": "section_id",
                "type": "Edm.String",
                "searchable": True
            },
            # Vector fields for embeddings
            {
                "name": "partVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": 1536,
                "vectorSearchProfile": "amlHnswProfile"
                
            },
            {
                "name": "chapterVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": 1536,
                "vectorSearchProfile": "amlHnswProfile"
            },
            {
                "name": "sectionVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": 1536,
                "vectorSearchProfile": "amlHnswProfile"
            },
            {
                "name": "keywordVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": 1536,
                "vectorSearchProfile": "amlHnswProfile"
            },
            {
                "name": "paraVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": 1536,
                "vectorSearchProfile": "amlHnswProfile"
            },
            {
                "name": "topicVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": 1536,
                "vectorSearchProfile": "amlHnswProfile"
            },
            {
                "name": "summaryVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": 1536,
                "vectorSearchProfile": "amlHnswProfile"
            },
            {
                "name": "categoryVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": 1536,
                "vectorSearchProfile": "amlHnswProfile"
            },
            # Existing fields such as lastUpdated, suggesters, scoringProfiles, etc.
        ],
        "scoringProfiles": [
            {
            "name": "paraboost",
            "text": {
                "weights": {
                f"para": 5
                }
            },
                "functions": []
            },
            {
            "name": "newAndLatest",
            "functionAggregation": "sum",
            "functions": [
                {
                    "fieldName": "lastUpdated",
                    "interpolation": "quadratic",
                    "type": "freshness",
                    "boost": 10,
                    "freshness": {
                            "boostingDuration": "P365D"
                        }
            
                }
            ]
            }
        ],
        "suggesters": [
            {
                "name": "sg",
                "searchMode": "analyzingInfixMatching",
                "sourceFields": ["section_title", "chapter_title"]
            }
        ],
        "vectorSearch": {
                "algorithms": [
                    {
                        "name": "amlHnsw",
                        "kind": "hnsw",
                        "hnswParameters": {
                        "m": 4,
                        "metric": "cosine"
                        }
                    }
                
                ],
                "profiles": [
                    {
                        "name": "amlHnswProfile",
                        "algorithm": "amlHnsw",
                        "vectorizer": "amlVectorizer"
                    }
                
                ], 
                "vectorizers": [
                    {
                        "name":"amlVectorizer",
                        "kind":"azureOpenAI",
                        "azureOpenAIParameters": {
                            "resourceUri": azure_openai_endpoint,
                            "deploymentId": azure_openai_embedding__large_deployment,
                            "modelName": embedding_model_name,
                            "apiKey": azure_openai_key
                        }
                    }
                ]
                
    },
        "semantic": {
            "configurations": [
                {
                    "name": "aml-semantic-config",
                    "prioritizedFields": {
                        "titleField": {
                            "fieldName": "section_title"
                        },
                        "prioritizedKeywordsFields": [
                            {
                                "fieldName": "category"
                            },
                            {
                                "fieldName": "topics"
                            }
                        ],
                        "prioritizedContentFields": [
                            {
                                "fieldName": f"para"
                            }
                        ]
                    }
                }
            ]
        }
    }



        headers = {'Content-Type': 'application/json',
                'api-key': os.getenv("AZURE_SEARCH_ADMIN_KEY", "") }
        # Create Index
        url = azure_search_endpoint + "/indexes/" + index_name + "?api-version=2024-07-01"


        response = requests.get(url, headers=headers)
        if response.status_code == 404:
            response  = requests.put(url, headers=headers, json=index_schema)
            index = response.json()
            print(index)
        else:
            print("Index already exists")

def get_document_layout(pdf_folder, doc_name):
    with open(os.path.join(pdf_folder ,doc_name), "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", analyze_request=f, content_type="application/octet-stream"
        )
    return poller.result()

def extract_pdf_data(pdf_folder, extract_folder):
     doc_names = [os.listdir(pdf_folder)[i] for i in range(0, len(os.listdir(pdf_folder)))]
     for doc_idx, doc_name in enumerate(doc_names):
        # Get the document layout
        document_data = []
        print(f"Analyzing document: {doc_name}")
        result = get_document_layout(pdf_folder, doc_name)
        print(f"Layout analysis completed for document: {doc_name}")
        print(f"Processing document: {doc_name}...")
        for page in result.pages:
            if page.lines:
                page_text = ""
                for line_idx, line in enumerate(page.lines):
                    #print(f"Line {line_idx}: {line.content}")
                    page_text +=  line.content + " "

                doc_data = {
                    "doc_name": doc_name,
                    "page_number": page.page_number,
                    "line_number": line_idx,
                    "content": page_text
                }
                document_data.append(doc_data)

        output_file_path = os.path.join(extract_folder, doc_names[doc_idx] + "-document_data.json")
        with open(output_file_path, "w") as f:
            json.dump(document_data, f)


enrichment_system_message = """
    You are an AI assitant who can extract title, topics and cateogries from a text document.
    You will be given a text document and you need to extract the title, topics, categories and summary from the document in json format.
    Topics: Extract the topics from the document that best describe the content.
    Categories: Extract the categories from the document that best describe the content.
    Summary: Extract the summary of the document.
    Do not write ```json and ``` in your response.

    json format:
    {
        "topics": ["topic1", "topic2"],
        "categories": ["category1", "category2"],
        "summary": "summary of the document"
    }
    """


processed_docs_count = 0
def process_jsonldoc(jsonldoc, system_message):
    try:
        doc = json.loads(jsonldoc)
        global processed_docs_count
        
        user_query = f"""Extract the topics, categories, and summary from the document.
        
        Document:
        
        {doc["para"]}
        """

        token_count = openai_helper.get_token_count(user_query)
        print(f"Processing Token count: {token_count}")

        llm_response = openai_helper.getOpenAIRespWithRetry(user_query, system_message)
        llm_json = json.loads(llm_response)
        processed_docs_count += 1
        print(f"Processed document: {doc.get('book_title', 'unknown')}, {doc.get('section_title', 'unknown')}, docs count: {processed_docs_count}")
        return {
            "id": str(uuid.uuid4()),
            "book_title": doc["book_title"],
            "part_title": doc.get("part_title", ""),
            "part_id": doc.get("part_id", ""),
            "chapter_title": doc.get("chapter_title", ""),
            "chapter_id": doc.get("chapter_id", ""),
            "section_title": doc["section_title"],
            "section_id": doc["section_id"],
            "keywords": json.dumps(doc["keywords"]),
            "xrefs": json.dumps(doc.get("xrefs", [])),
            "para": doc["para"],
            "topics": json.dumps(llm_json["topics"]),
            "summary": llm_json["summary"],
            "category": json.dumps(llm_json["categories"]),
            "lastupdated": str(datetime.now())
        }
        
    except Exception as e:
        with open("error.log", "a") as error_file:
            error_file.write(f"Error processing document: {doc.get('book_title', 'unknown')}, {doc.get('section_title', 'unknown')} - {e}\n")
        print(f"Error processing document: {doc.get('book_title', 'unknown')}, {doc.get('section_title', 'unknown')} - {e}")
        return None  # Return None in case of an error to filter later


def process_single_document(file_path, system_message):
    alta_index_data = []
    with open(file_path, "r") as f:
        alta_docs_json = json.loads(f.read())
        
        # Use ThreadPoolExecutor for parallel processing of each jsonldoc
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(process_jsonldoc, jsonldoc, system_message) for jsonldoc in alta_docs_json]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    alta_index_data.append(result)

    return alta_index_data


def enrich_pdf_data_parallel(extracted_data_folder, output_file_name):
    alta_index_data = []
    files = [os.path.join(extracted_data_folder, f) for f in os.listdir(extracted_data_folder)]
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_document, file, enrichment_system_message): file for file in files}
        for future in as_completed(futures):
            try:
                alta_index_data.extend(future.result())
            except Exception as e:
                print(f"Error processing file: {futures[future]} - {e}")

    with open(output_file_name, "w") as output_file:
        json.dump(alta_index_data, output_file)

def enrich_pdf_data(extracted_data_folder, output_file_name):
    alta_index_data = []
    
    for ex_data in os.listdir(extracted_data_folder):
        #print(f"Processing extracted data: {ex_data}")
        with open(os.path.join(extracted_data_folder, ex_data), "r") as f:
            alta_docs_json = json.loads( f.read())
            print(f"Processing document: {f.name}")
            item_count = 0
            for jsonldoc in alta_docs_json:
                doc = json.loads(jsonldoc)
                #print(f"Processing document: {doc['doc_name']}")
                user_query = f"""Extract the topics, categories and summary from the document.
                            
                            Document:

                            {doc["para"]}
                            """
                try:
                    llm_reponse =openai_helper.getOpenAIResp(user_query, enrichment_system_message)
                    llm_json = json.loads(llm_reponse)
                    aml_index_item = {
                        "id": str(uuid.uuid4()),
                        "book_title": doc["book_title"],
                        "part_title": doc["part_title"],
                        "part_id": doc["part_id"],
                        "chapter_title": doc["chapter_title"],
                        "chapter_id": doc["chapter_id"],
                        "section_title": doc["section_title"],
                        "section_id": doc["section_id"],
                        "keywords": json.dumps(doc["keywords"]),
                        "para": doc["para"],
                        "topics": json.dumps(llm_json["topics"]),
                        "summary": llm_json["summary"],
                        "category": json.dumps(llm_json["categories"]),
                        "lastupdated": str(datetime.now())
                    }

                    alta_index_data.append(aml_index_item)
                    item_count += 1
                    print(f"Processed document: {doc['book_title']}, {doc['section_title']} - {item_count}")
                except Exception as e:
                    with open("error.log", "a") as f:
                        f.write(f"Error processing document: {doc['book_title']}, {doc['section_title']} - {e}\n")
                    print(f"Error processing document: {doc['book_title']}, {doc['section_title']} - {e}")

    with open(output_file_name, "w") as f:
        json.dump(alta_index_data, f) 
        


def generate_embeddings_for_doc1(doc):
    # Extract fields
    try:
        global processed_docs_count
        part_title = doc.get("part_title", "")
        chapter_title = doc.get("chapter_title", "")
        section_title = doc["section_title"]
        keywords = json.loads(doc.get("keywords", "[]"))
        para = doc["para"]
        topics = json.loads(doc.get("topics", "[]"))
        summary = doc.get("summary", "")
        category = json.loads(doc.get("category", "[]"))
        
        # Generate embeddings for each part, chapter, section, keywords, para, topics, summary, category
        part_embeddings = openai_helper.generate_embeddings(
            part_title, dimensions=azure_openai_embedding_small_dimensions,
            model=azure_openai_embedding__small_deployment
        )
        chapter_embeddings = openai_helper.generate_embeddings(
            chapter_title, dimensions=azure_openai_embedding_small_dimensions,
            model=azure_openai_embedding__small_deployment
        )
        section_embeddings = openai_helper.generate_embeddings(
            section_title, dimensions=azure_openai_embedding_small_dimensions,
            model=azure_openai_embedding__small_deployment
        )
        keyword_embeddings = openai_helper.generate_embeddings(
            keywords, dimensions=azure_openai_embedding_small_dimensions,
            model=azure_openai_embedding__small_deployment
        )
        para_embeddings = openai_helper.generate_embeddings(
            para, dimensions=azure_openai_embedding_large_dimensions,
            model=azure_openai_embedding__large_deployment
        )
        topic_embeddings = openai_helper.generate_embeddings(
            topics, dimensions=azure_openai_embedding_small_dimensions,
            model=azure_openai_embedding__small_deployment
        )
        summary_embeddings = openai_helper.generate_embeddings(
            summary, dimensions=azure_openai_embedding_small_dimensions,
            model=azure_openai_embedding__small_deployment
        )
        category_embeddings = openai_helper.generate_embeddings(
            category, dimensions=azure_openai_embedding_small_dimensions,
            model=azure_openai_embedding__small_deployment
        )
        
        # Update document with embeddings
        doc["partVector"] = part_embeddings[0].embedding if part_embeddings else None
        doc["chapterVector"] = chapter_embeddings[0].embedding if chapter_embeddings else None
        doc["sectionVector"] = section_embeddings[0].embedding if section_embeddings else None
        doc["keywordVector"] = keyword_embeddings[0].embedding if keyword_embeddings else None
        doc["paraVector"] = para_embeddings[0].embedding if para_embeddings else None
        doc["topicVector"] = topic_embeddings[0].embedding if topic_embeddings else None
        doc["summaryVector"] = summary_embeddings[0].embedding if summary_embeddings else None
        doc["categoryVector"] = category_embeddings[0].embedding if category_embeddings else None
        
        processed_docs_count += 1
        print(f"Processed document: {doc.get('book_title', 'unknown')}, {doc.get('section_title', 'unknown')} - {processed_docs_count}")
        return doc
    except Exception as e:
        with open("embedding_error.log", "a") as f:
            f.write(str(doc))
        print(f"generate_embeddings_for_doc Error processing document: {e}")
        return None


import time
import json



def generate_embeddings_for_doc_simple(doc, max_retries=5, base_delay=15):
    # Extract fields
    try:
        global processed_docs_count
        content = doc.get("content", "")
        

        # Function to attempt embedding generation with retry logic
        def generate_with_retries(data, dimensions, model):
            retries = 0
            while retries < max_retries:
                try:
                    if type(data) == list:
                        cleaned_data = [item for item in data if item is not None]
                        cleaned_data_string = ", ".join(cleaned_data)
                        return openai_helper.generate_embeddings(cleaned_data_string, dimensions=dimensions, model=model)
                    else:
                        return openai_helper.generate_embeddings(data, dimensions=dimensions, model=model)

                except Exception as e:
                    retries += 1
                    delay = base_delay * (2 ** (retries - 1))  # Exponential backoff
                    print(f"Retry {retries} for {data}. Waiting {delay} seconds before retrying...")
                    time.sleep(delay)
            print(f"Failed to generate embeddings after {max_retries} attempts.")
            return None

        # Generate embeddings for each part, chapter, section, keywords, para, topics, summary, category
        content_embeddings = generate_with_retries(content, azure_openai_embedding_small_dimensions, azure_openai_embedding__small_deployment)
        
        # Update document with embeddings
        doc["contentVector"] = content_embeddings[0].embedding if content_embeddings else None
        processed_docs_count += 1
        print(f"Processed document: {doc.get('id', 'unknown')} - {processed_docs_count}")
        return doc
    except Exception as e:
        with open("embedding_error.log", "a") as f:
            f.write(str(doc) + "\n")
        print(f"generate_embeddings_for_doc Error processing document: {e}")
        return None

def generate_embeddings_for_doc(doc, max_retries=5, base_delay=15):
    # Extract fields
    try:
        global processed_docs_count
        part_title = doc.get("part_title", "")
        chapter_title = doc.get("chapter_title", "")
        section_title = doc["section_title"]
        keywords = json.loads(doc.get("keywords", "[]"))
        para = doc["para"]
        topics = json.loads(doc.get("topics", "[]"))
        summary = doc.get("summary", "")
        category = json.loads(doc.get("category", "[]"))

        # Function to attempt embedding generation with retry logic
        def generate_with_retries(data, dimensions, model):
            retries = 0
            while retries < max_retries:
                try:
                    if type(data) == list:
                        cleaned_data = [item for item in data if item is not None]
                        cleaned_data_string = ", ".join(cleaned_data)
                        return openai_helper.generate_embeddings(cleaned_data_string, dimensions=dimensions, model=model)
                    else:
                        return openai_helper.generate_embeddings(data, dimensions=dimensions, model=model)

                except Exception as e:
                    retries += 1
                    delay = base_delay * (2 ** (retries - 1))  # Exponential backoff
                    print(f"Retry {retries} for {data}. Waiting {delay} seconds before retrying...")
                    time.sleep(delay)
            print(f"Failed to generate embeddings after {max_retries} attempts.")
            return None

        # Generate embeddings for each part, chapter, section, keywords, para, topics, summary, category
        part_embeddings = generate_with_retries(part_title, azure_openai_embedding_small_dimensions, azure_openai_embedding__small_deployment)
        chapter_embeddings = generate_with_retries(chapter_title, azure_openai_embedding_small_dimensions, azure_openai_embedding__small_deployment)
        section_embeddings = generate_with_retries(section_title, azure_openai_embedding_small_dimensions, azure_openai_embedding__small_deployment)
        keyword_embeddings = generate_with_retries(keywords, azure_openai_embedding_small_dimensions, azure_openai_embedding__small_deployment)
        para_embeddings = generate_with_retries(para, azure_openai_embedding_small_dimensions, azure_openai_embedding__small_deployment)
        topic_embeddings = generate_with_retries(topics, azure_openai_embedding_small_dimensions, azure_openai_embedding__small_deployment)
        summary_embeddings = generate_with_retries(summary, azure_openai_embedding_small_dimensions, azure_openai_embedding__small_deployment)
        category_embeddings = generate_with_retries(category, azure_openai_embedding_small_dimensions, azure_openai_embedding__small_deployment)

        # Update document with embeddings
        doc["partVector"] = part_embeddings[0].embedding if part_embeddings else None
        doc["chapterVector"] = chapter_embeddings[0].embedding if chapter_embeddings else None
        doc["sectionVector"] = section_embeddings[0].embedding if section_embeddings else None
        doc["keywordVector"] = keyword_embeddings[0].embedding if keyword_embeddings else None
        doc["paraVector"] = para_embeddings[0].embedding if para_embeddings else None
        doc["topicVector"] = topic_embeddings[0].embedding if topic_embeddings else None
        doc["summaryVector"] = summary_embeddings[0].embedding if summary_embeddings else None
        doc["categoryVector"] = category_embeddings[0].embedding if category_embeddings else None

        processed_docs_count += 1
        print(f"Processed document: {doc.get('book_title', 'unknown')}, {doc.get('section_title', 'unknown')} - {processed_docs_count}")
        return doc
    except Exception as e:
        with open("embedding_error.log", "a") as f:
            f.write(str(doc) + "\n")
        print(f"generate_embeddings_for_doc Error processing document: {e}")
        return None


def enrich_with_embeddings_parallel_simple(output_file_name):
    # Load data from the JSON file
    with open(output_file_name, "r") as f:
        search_index_data = json.loads(f.read())

    # Parallel processing with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_embeddings_for_doc_simple, doc) for doc in search_index_data]
        results = []
        
        # As each future completes, add the result to `results`
        for future in as_completed(futures):
            try:
                fresult = future.result()
                if fresult is not None:
                    results.append(fresult)
            except Exception as e:
                with open("fresult_error.log", "a") as f:
                    f.write(str(fresult))
                print(f"Error processing document: {e}")
    
    # Save the enriched data with embeddings
    with open(f"{output_file_name}_with_vectors.json", "w") as f:
        f.write(json.dumps(results))

def enrich_with_embeddings_parallel(output_file_name):
    # Load data from the JSON file
    with open(output_file_name, "r") as f:
        search_index_data = json.loads(f.read())

    # Parallel processing with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_embeddings_for_doc, doc) for doc in search_index_data]
        results = []
        
        # As each future completes, add the result to `results`
        for future in as_completed(futures):
            try:
                fresult = future.result()
                if fresult is not None:
                    results.append(fresult)
            except Exception as e:
                with open("fresult_error.log", "a") as f:
                    f.write(str(fresult))
                print(f"Error processing document: {e}")
    
    # Save the enriched data with embeddings
    with open(f"{output_file_name}_with_vectors.json", "w") as f:
        f.write(json.dumps(results))



def upload_to_search_simple(index_name, data_file, language_suffix: str = "en"):
    
    vector_file_name = f"{data_file}_with_vectors.json"

    with open(vector_file_name, "r") as f:
        aml_index_data_with_vectors = json.loads(f.read())

    search_client = SearchClient(endpoint=azure_search_endpoint, index_name=index_name, credential=credential)

    for idx, doc in enumerate(aml_index_data_with_vectors):
        search_doc = {
            "id": doc["id"],
            "content": doc.get("content", ""),
            "contentVector": doc.get("contentVector") or [],
        }

        result = search_client.upload_documents(documents=[search_doc])
        print(f"Uploaded document: {doc['id']} - {idx + 1}")

    print(f"{len(aml_index_data_with_vectors)} Documents uploaded to Azure Search")


    





def upload_to_search(index_name, data_file, language_suffix: str = "en"):
    
    vector_file_name = f"{data_file}_with_vectors.json"

    with open(vector_file_name, "r") as f:
        aml_index_data_with_vectors = json.loads(f.read())

    search_client = SearchClient(endpoint=azure_search_endpoint, index_name=index_name, credential=credential)

    for idx, doc in enumerate(aml_index_data_with_vectors):
        
        last_updated = datetime.fromisoformat(doc["lastupdated"]).isoformat() + "Z"
        search_doc = {
            "id": doc["id"],
            "part_title": doc.get("part_title", ""),
            "chapter_title": doc["chapter_title"],
            "section_title": doc["section_title"],
            "part_id": doc.get("part_id", ""),
            "chapter_id": doc.get("chapter_id", ""),
            "section_id": doc.get("section_id", ""),
            "keywords": [keyword for keyword in json.loads(doc["keywords"]) if keyword is not None],
            "para": doc["para"],
            "topics": [topic for topic in json.loads(doc["topics"]) if topic is not None],
            "summary": doc["summary"],
            "category": [category for category in json.loads(doc["category"]) if category is not None],
            "lastUpdated": last_updated,
            "partVector": doc.get("partVector") or [],
            "chapterVector": doc.get("chapterVector") or [],
            "sectionVector": doc.get("sectionVector") or [],
            "keywordVector":  doc.get("keywordVector") or [],
            "paraVector": doc.get("paraVector") or [],
            "topicVector": doc.get("topicVector") or [],
            "summaryVector": doc.get("summaryVector") or [],
            "categoryVector": doc.get("categoryVector") or [],

        }

        result = search_client.upload_documents(documents=[search_doc])
        print(f"Uploaded document: {doc['section_title']} - {idx + 1}")

    print(f"{len(aml_index_data_with_vectors)} Documents uploaded to Azure Search")


def get_index_fields(index_name):
    index_client = SearchIndexClient(
        endpoint=azure_search_endpoint, credential=credential)
    idx = index_client.get_index(index_name)
    select_fields = []
    vector_fields =  []
    for field in idx.fields:
        #print(field.name)
        if(field.type == SearchFieldDataType.String):
            select_fields.append(field.name)
        if(str.find(field.name, "Vector") > 0):
            vector_fields.append(field.name)
    return select_fields, vector_fields

async def retrieve_search_results(index_name: str, search_query: str, top_k: int = 10) -> str:
    select_fields, vector_fields = get_index_fields(index_name)  
    vector_fields = ["sectionVector", "chapterVector", "partVector", "paraVector", "topicVector", "summaryVector"]
    #select_fields = ["title", "content", "category", "tags"]
    search_client = SearchClient(endpoint=azure_search_endpoint, index_name=index_name, credential=credential)
    #vector_query = VectorizableTextQuery(text=search_query, k_nearest_neighbors=3, fields=search_fields, exhaustive=True)
  
    vector_queries  = [VectorizableTextQuery(text=search_query, k_nearest_neighbors=top_k, fields=field, exhaustive=True) for field in vector_fields]
    results = search_client.search(  
        search_text=search_query,  
        vector_queries= vector_queries,
        select=select_fields,
        top=top_k
    )  

    json_results = []
    for result in results:
        field_results = []
        for field in select_fields:
            result_dict = {
                field: result[field]
            }
            field_results.append(result_dict)
        json_results.append(field_results)
    #print(json_results)
    return json_results
    #return f"<Context>{ json.dumps(json_results)} </Context>"