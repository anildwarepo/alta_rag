import search_helper_alta
import os
import openai_helper
import json
import datetime
import uuid
import asyncio


def create_json_file(doc_source_folder):
    
    # list file in fintech folder
    source_docs_json = []

    for file in os.listdir(doc_source_folder):
        print(file)
        if file.endswith(".txt"):
            print(f"Processing file: {file}")
            with open(f"{doc_source_folder}/{file}", "r") as f:
                content = f.read()
                json_doc = {
                    "id": str(uuid.uuid4()),
                    "content": content
                }
                source_docs_json.append(json_doc)
    
    with open("fintech_kb.json", "w") as f:
        f.write(json.dumps(source_docs_json))

if __name__ == "__main__":

    language_suffix = "en"
    analyzer_name = "en.microsoft"
    index_name = "alta_index"

    doc_source_folder = "fintech-kb-articles"
    #create_json_file(doc_source_folder)
                

    doc_source_json_folder = "fintech-docs-extracted"

    #search_helper_alta.create_simple_index(index_name, analyzer_name=analyzer_name, language_suffix=language_suffix)

    #search_helper.extract_pdf_data("german_document", "german_document_extracted_data")

    #extracted_data_folder = "alta_extracted_data"
    #output_file_name = os.path.join("fintech-docs-extracted", "fintech_kb.json")
    #search_helper_alta.enrich_pdf_data_parallel(doc_source_json_folder, output_file_name)
    
    #search_helper_alta.enrich_with_embeddings_parallel_simple(output_file_name)

    output_file_name = "alta_enriched_data.json"
    search_helper_alta.create_index(index_name)
    vector_file_name = f"{output_file_name}_with_vectors.json"
    with open(vector_file_name, "r") as f:
        aml_index_data_with_vectors = json.loads(f.read())
    #print(len(aml_index_data_with_vectors))
    
    search_helper_alta.upload_to_search(index_name, output_file_name, language_suffix=language_suffix)
    #search_helper_alta.upload_to_search_simple(index_name, output_file_name, language_suffix=language_suffix)


    #index_name = "alta_index_with_suggester"
    #search_query = "what are the steps for installing netbackup for DB2 and explain each step in detail?"
    #async def run_search():
    #    sr = await search_helper_alta.retrieve_search_results(index_name=index_name, search_query=search_query)
    #    print(sr)
    #
    #
    #asyncio.run(run_search())