"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-11-13 11:06:44
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-11-15 18:25:21
FilePath: scripts/DataLoader.py
Description: Loading data functions in this file
"""

import json


def read_linked_docred(linked_docred_path: str) -> dict:
    with open(linked_docred_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    return dataset

"""
The function return a list of documents.
Each documents is a list of sentences
Each sentences is a list of tokens = {name:__}
** After Knowledge Extract, tokens should additionally have 'type' = {name:__, type:__} **
"""
def loading_sentences(file_path: str) -> list:
    
    linked_docred = read_linked_docred(file_path)

    documents = []

    for document in linked_docred:
        documents.append(document['sents'])
        # sents = document['sents']
        # 
        # doc_sents = []
        # 
        # for sent in sents:
        #     tokens = []
        #     for token in sent:
        #         tokens.append({'name': token})
        #     doc_sents.append({'entities': tokens})
        # 
        # documents.append({
        #     'title': document.get('title', ''),
        #     'sents': doc_sents
        # })
    
    return documents


"""
Store the list of the documents with json format
Each documents is a list of sentences
Each list contains the ground truth entities = {name:__, type:__}
"""
def making_entity_gt(file_path: str, output_file: str):
    linked_docred = read_linked_docred(file_path)

    documents = []

    for document in linked_docred:
        sents = document['sents']
        entities = document['entities']

        doc_entities = []

        for sent_id, sent in enumerate(sents):
            labels = ["O"] * len(sent)
            token_labels = []

            for entity in entities:
                for mention in entity['mentions']:
                    if mention['sent_id'] == sent_id:
                        start, end = mention['pos']
                        entity_type = entity['type']

                        # Apply BIO tagging ex. beginning word = B-ORG, inside word = I-ORG, else = O
                        labels[start] = f"B-{entity_type}"
                        for i in range(start + 1, end):
                            labels[i] = f"I-{entity_type}"

            for token, label in zip(sent, labels):
                token_labels.append({'name':token, 'type':label})
            
            doc_entities.append({'entities':token_labels})

        documents.append({
            'title': document.get('title', ''),
            'sents': doc_entities
        })

    print(documents[0])

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=4, ensure_ascii=False)

    print(f"Entity ground truth save to {output_file}")



def making_graph_gt(file_path: str, output_file: str):
    linked_docred = read_linked_docred(file_path)

    # knowledge graph list for each docs.
    kg_data = []

    for document in linked_docred:
        doc_title = document.get("title", "")
        entities = document['entities']
        relations = document['relations']

        # getting node info. (whole entities from 1 doc)
        nodes = []
        for entity in entities:
            entity_id = entity['id']
            entity_type = entity['type']
            mentions = entity['mentions']

            entity_info = {
                "id": entity_id,
                "type": entity_type,
                "mentions" : mentions
            }
            nodes.append(entity_info)

        # getting edge info. (whole relations from 1 doc)
        edges = []
        for relation in relations:
            head_id = relation['h']
            tail_id = relation['t']
            relation_type = relation['r']
            evidence = relation['evidence']

            edge_info = {
                "relation_type": relation_type,
                "head": head_id,
                "tail": tail_id,
                "evidence": evidence
            }
            edges.append(edge_info)

        # store Knowledge Graph structure
        kg_entry = {
            "title": doc_title,
            "nodes": nodes,
            "edges": edges
        }

        kg_data.append(kg_entry)

    print(kg_data[0])

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, indent=4)

    print(f"Graph ground truth save to {output_file}")
    
import json

def read_extracted_entities_json(filepath: str) -> dict:
    """
    Reads a JSON file containing extracted entities and their metadata and returns its content as a dictionary.

    :param filepath: Path to the JSON file.
    :return: A dictionary where the keys are `doc_index` and the values are the entities dictionary for each document.
    """
    try:
        # Open and load the JSON file
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Convert the list of documents into a dictionary indexed by `doc_index`
        extracted_entities = {}
        for entry in data:
            if 'doc_index' in entry and 'entities' in entry:
                extracted_entities[entry['doc_index']] = entry['entities']
            else:
                raise ValueError(f"Malformed entry in JSON file: {entry}")

        return extracted_entities

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {filepath} was not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode JSON from the file at {filepath}. Please ensure it is correctly formatted.")

    


#making_entity_gt('dataset_Linked-DocRED/test.json', 'dataset_gt/test_NER_gt.json')
#making_graph_gt('dataset_Linked-DocRED/dev.json', 'dataset_gt/dev_GRAPH_gt.json')