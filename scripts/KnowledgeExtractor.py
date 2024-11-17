"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-11-13 11:13:30
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-11-15 20:52:04
FilePath: scripts/KnowledgeExtractor.py
Description: functions to extract entities and linking between entitites to create a knowledge graph
"""
DEBUG = 1

import json
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForTokenClassification
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
from litellm import completion

from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

tokenizer_for_relationship = T5Tokenizer.from_pretrained("google/flan-t5-large")
model_for_relationship = T5ForConditionalGeneration.from_pretrained("knowledgator/t5-for-ie").to(device)


debug_sentences = [
    ["Elon", "Musk", "is", "the", "CEO", "of", "Tesla", "and", "SpaceX"],
    ["Jeff", "Bezos", "founded", "Amazon", "and", "he", "owns", "The", "Washington", "Post"],
    ["Microsoft", "released", "Windows", "10", "under", "the", "leadership", "of", "Satya", "Nadella"],
    ["Tim", "Cook", "introduced", "the", "iPhone", "12", "during", "an", "Apple", "event"],
    ["Mark", "Zuckerberg", "announced", "the", "rebranding", "of", "Facebook", "to", "Meta"],
    ["Bill", "Gates", "and", "Paul", "Allen", "founded", "Microsoft", "in", "1975"],
    ["Steve", "Jobs", "presented", "the", "first", "iPad", "in", "2010"],
    ["Sundar", "Pichai", "is", "the", "CEO", "of", "Google", "and", "Alphabet"],
    ["Sheryl", "Sandberg", "served", "as", "COO", "at", "Facebook", "and", "wrote", "Lean", "In"],
    ["The", "Coca-Cola", "Company", "produces", "Coke", "and", "Sprite"],
    ["Nike", "introduced", "the", "Air", "Jordan", "series", "in", "collaboration", "with", "Michael", "Jordan"],
    ["Henry", "Ford", "founded", "Ford", "Motor", "Company", "and", "revolutionized", "manufacturing"],
    ["Jack", "Ma", "started", "Alibaba", "and", "owns", "a", "stake", "in", "Ant", "Group"],
    ["Larry", "Page", "and", "Sergey", "Brin", "co-founded", "Google", "while", "at", "Stanford"],
    ["Howard", "Schultz", "expanded", "Starbucks", "into", "a", "global", "brand"],
    ["Richard", "Branson", "launched", "Virgin", "Galactic", "for", "commercial", "space", "travel"],
    ["Toyota", "introduced", "the", "Prius", "as", "the", "world's", "first", "mass-produced", "hybrid", "car"],
    ["Oprah", "Winfrey", "launched", "OWN", "and", "partnered", "with", "Weight", "Watchers"],
    ["Warren", "Buffett", "is", "the", "CEO", "of", "Berkshire", "Hathaway"],
    ["Larry", "Ellison", "co-founded", "Oracle", "Corporation"],
    ["Sony", "released", "the", "PlayStation", "5", "in", "2020"],
    ["Amazon", "introduced", "the", "Kindle", "e-reader", "in", "2007"],
    ["Elon", "Musk", "introduced", "the", "Tesla", "Model", "3", "as", "an", "affordable", "electric", "car"],
    ["IBM", "developed", "Watson", "to", "compete", "in", "AI", "and", "machine", "learning"],
    ["Apple", "introduced", "the", "MacBook", "Air", "under", "Steve", "Jobs'", "leadership"],
    ["Diane", "von", "Furstenberg", "created", "the", "iconic", "wrap"]
]


def parse_input_sentences(document):
    """
    Transforms the input document into a list of sentences, where each sentence is a list of token strings.

    :param document: A dictionary containing the document structure:
                     {
                        'title': <document title>,
                        'sents': [
                            {'entities': [{'name': <token>}, ...]}, 
                            ...
                        ]
                     }
    :return: A list of sentences, where each sentence is a list of token strings.
             Example: [["token1", "token2", ...], ["tokenA", "tokenB", ...], ...]
    """
    transformed_sentences = []

    # Check if 'sents' key exists in the document
    if 'sents' not in document or not isinstance(document['sents'], list):
        raise ValueError("Invalid document format: 'sents' key is missing or not a list.")

    # Iterate over the sentences in the document
    for sentence_data in document['sents']:
        if 'entities' in sentence_data and isinstance(sentence_data['entities'], list):
            # Extract token names for each sentence
            sentence_tokens = [token['name'] for token in sentence_data['entities'] if 'name' in token]
            transformed_sentences.append(sentence_tokens)
        else:
            raise ValueError("Invalid sentence format: 'entities' key is missing or not a list.")

    return transformed_sentences



def store_entities_per_document(doc_index, entities):
    """
    Store entities per document in a JSON file.

    :param doc_index: Index of the document.
    :param entities: Dictionary of entities in the format:
                     {
                        entity.text: [
                            entity.type,
                            [ref index list to sentences that mention the entity]
                        ]
                     }
    """
    output_file = "extracted_entities_bert_big.json"

    # Initialize the data structure to be stored
    document_data = {
        "doc_index": doc_index,
        "entities": entities
    }

    # Check if the file already exists
    if os.path.exists(output_file):
        # Load existing data
        with open(output_file, "r") as f:
            existing_data = json.load(f)
    else:
        # Start with an empty list if the file doesn't exist
        existing_data = []

    # Add the new document data
    existing_data.append(document_data)

    # Write the updated data back to the file
    with open(output_file, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Entities for document {doc_index} stored in {output_file}.")

def store_relationships_per_document(doc_index, relationships):
    """
    Store relationships per document in a JSON file.

    :param doc_index: Index of the document.
    :param relationships: Dictionary of relationships in the format:
                          {
                              "relationship_id": [
                                  "entity1.text",
                                  "entity2.text",
                                  "relationship_label",
                                  sentence_index
                              ]
                          }
    """
    output_file = "extracted_relationship_big.json"
    output_dir = os.path.dirname(output_file)

    # Ensure the directory for the output file exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Initialize the data structure to be stored
    document_data = {
        "doc_index": doc_index,
        "relationships": relationships
    }

    # Check if the file already exists
    if os.path.exists(output_file):
        # Load existing data
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        # Start with an empty list if the file doesn't exist
        existing_data = []

    # Add the new document data
    existing_data.append(document_data)

    # Write back the updated data to the JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)


def extract_knowledge(doc_index, document, tagger):
    """
    Extracts entities and relationships from a document represented as a list of tokenized sentences.

    *Functionality*:
    1. **Entity Extraction**:
       - Uses the Flair NER tagger to identify named entities (e.g., people, locations) in each sentence.
       - Stores entities in a global dictionary (`global_entities_dict`) with their type (label) and references to sentence indices where they occur.
       - Handles duplicate entities by aggregating sentence indices for previously identified entities.

    2. **Relationship Extraction** (Placeholder):
       - This step is designed to infer relationships between entities using an LLM.
       - Currently, it is not implemented but sets up the structure for future expansion.

    3. **Data Persistence**:
       - Saves extracted entities to a JSON file (`extracted_entities_flair.json`) using the `store_entities_per_document` function.
       - The output JSON contains the document index and the list of entities with their types and references.

    **Parameters**:
    - `doc_index` (int): 
        Index of the document being processed. Used for identifying the document in the JSON file.
    - `sentences` (list of list of str): 
        A list of tokenized sentences. Each sentence is a list of word tokens (e.g., `[['Alice', 'went'], ['Bob', 'lives']]`).
    - `tagger` (flair.models.SequenceTagger): 
        Pre-loaded Flair NER tagger model for entity recognition.

    **Global Dictionary Format**:
    - `global_entities_dict`: Dictionary with the following structure:
      ```python
      {
          entity_text: [entity_label, [sentence_indices]]
      }
      ```
      - `entity_text` (str): The exact text of the entity (e.g., "Alice").
      - `entity_label` (str): The type of the entity (e.g., "PER" for person, "LOC" for location).
      - `sentence_indices` (list of int): List of indices of sentences where the entity is mentioned.

    **JSON Output Format**:
    - The JSON file (`extracted_entities_flair.json`) will contain:
      ```json
      [
          {
              "doc_index": <doc_index>,
              "entities": {
                  "<entity_text>": ["<entity_label>", [<sentence_indices>]]
              }
          }
      ]
      ```

    **Return Values**:
    - `global_entities_dict` (dict): Dictionary containing extracted entities.
    - `relationships` (list): Placeholder list for relationships (currently empty).

    **Steps**:
    1. **Initialize Variables**:
       - Create an empty `global_entities_dict` to store entities.
    2. **Entity Extraction**:
       - Loop through each sentence, reconstruct it from tokens, and run Flair NER.
       - Extract entities and their labels from the Flair prediction.
       - Check for duplicate entities and update their sentence indices if they are already in the dictionary.
    3. **Save Entities**:
       - Call `store_entities_per_document(doc_index, global_entities_dict)` to save the extracted entities to a JSON file.
    4. **Relationship Extraction**:
       - Set up a placeholder for extracting relationships between entities.

    **Example Usage**:
    ```python
    sentences = [
        ["Alice", "went", "to", "the", "store", "in", "New", "York", "."],
        ["Bob", "lives", "in", "San", "Francisco", "."]
    ]

    tagger = SequenceTagger.load('de-ner-large')  # Load Flair NER tagger
    doc_index = 1

    # Extract knowledge
    entities, relationships = extract_knowledge(doc_index, sentences, tagger)
    ```

    **Debugging**:
    - If `DEBUG` is set to `True`, the function prints:
      - Initial debug information about the sentences.
      - Number of entities extracted and the progress of processing.

    **Limitations**:
    - The current entity matching is exact (string-based). Entities with slight variations (e.g., "Greek" vs. "Greece") are treated as different.
    - Relationship extraction is not yet implemented.
    """
    
    global_entities_dict = {}  # Dictionary to store all entities with their labels and sentence references
    
    relationships = {}  # Dictionary to store all relationship with their labels and sentence references
    """
    relationships structure:
        {
            rel_id : [ # the id is sequential
                "entity.text",  # first entity
                "entity.text",  # second entity 
                "label", # description of the relationship
                "id of the sentences
            ]
            ...
        }
                    
    """
    # sentences = parse_input_sentences(document)
    print(document)
    sentences = document
    
    if DEBUG:
        print("Initiating the entity and relationship extraction process...")
        print(f"Processing document {doc_index} with {len(sentences)} sentences.")
    
    #id_relationships = 0

    # Step 1: Entity Extraction
    for index, sentence in enumerate(sentences):
        sentence_text = " ".join(sentence)  # Reconstruct the sentence from tokens
        flair_sentence = Sentence(sentence_text)

        # Run Flair NER on the sentence
        tagger.predict(flair_sentence)
        local_entities = flair_sentence.get_spans("ner")

        for i, entity in enumerate(local_entities):
            entity_text = entity.text.strip()  # Extract and clean the entity text
            entity_label = entity.get_label("ner").value  # Extract the entity label

            # Check if the entity is already in the global dictionary
            if entity_text not in global_entities_dict:
                # Add new entity with its label and sentence reference
                global_entities_dict[entity_text] = [entity_label, [index]]
            else:
                # Append the sentence index to the existing entity
                global_entities_dict[entity_text][1].append(index)
                        
        
    # Store extracted entities in the JSON file
    store_entities_per_document(doc_index, global_entities_dict)

    # Step 2: Relationship Extraction
    for index, sentence in enumerate(sentences):
        sentence_text = " ".join(sentence)  # Reconstruct the sentence from tokens
        flair_sentence = Sentence(sentence_text)

        # Run Flair NER on the sentence
        tagger.predict(flair_sentence)
        local_entities = flair_sentence.get_spans("ner")

        # Initialize a dictionary to store relationships in the current sentence
        local_relationships = {}

        # Check if there are multiple entities in the sentence
        if len(local_entities) > 1:
            for i, entity1 in enumerate(local_entities):
                for j, entity2 in enumerate(local_entities):
                    if i >= j:
                        continue  # Avoid duplicate relationships (A->B and B->A)

                    # Extract and clean entity texts
                    entity1_text = entity1.text.strip()
                    entity2_text = entity2.text.strip()

                    # Define a relationship label (can be improved with a language model)
                    relationship_label = "Related_to"

                    # Add the relationship to the local dictionary
                    relationship_id = f"rel_{len(local_relationships)}"
                    local_relationships[relationship_id] = {
                        "entity1": entity1_text,
                        "entity2": entity2_text,
                        "relationship": relationship_label,
                        "sentence_index": index
                    }

        # Append the local relationships to the global dictionary
        for relationship_id, relationship_data in local_relationships.items():
            relationships[relationship_id] = relationship_data

    # Store extracted relationships in the JSON file
    store_relationships_per_document(doc_index, relationships)



def extract_knowledge_bert(doc_index, document, model, tokenizer):
    """
    Extracts entities and relationships from a document represented as a list of tokenized sentences.
    This version uses a BERT-based model for entity extraction.

    Parameters:
    - `doc_index` (int): Index of the document being processed.
    - `document` (list of list of str): Tokenized sentences (e.g., [['Alice', 'went'], ['Bob', 'lives']]).
    - `model_name` (str): Name of the pre-trained Hugging Face model for NER.

    Returns:
    - `global_entities_dict` (dict): Extracted entities in the format:
      {
          "entity_text": ["entity_label", [sentence_indices]]
      }
    - `relationships` (list): Placeholder for relationships (currently empty).
    """
    global_entities_dict = {}  # Stores entities with labels and sentence references
    relationships = {}  # Placeholder for relationships

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    if DEBUG:
        print("Initiating the entity and relationship extraction process...")
        print(f"Processing document {doc_index} with {len(document)} sentences.")

    # Entity Extraction
    for index, sentence in enumerate(document):
        sentence_text = " ".join(sentence)  # Reconstruct the sentence from tokens

        # Predict entities using BERT NER pipeline
        ner_results = ner_pipeline(sentence_text)

        for entity in ner_results:
            entity_text = entity["word"].strip()
            entity_label = entity["entity_group"]

            # Check if the entity is already in the global dictionary
            if entity_text not in global_entities_dict:
                # Add new entity with its label and sentence reference
                global_entities_dict[entity_text] = [entity_label, [index]]
            else:
                # Append the sentence index to the existing entity
                global_entities_dict[entity_text][1].append(index)

    # Store extracted entities in the JSON file
    store_entities_per_document(doc_index, global_entities_dict)

    # Relationship Extraction
   
    relationships = {}  # Global dictionary to store relationships

    # Iterate through the document sentences
    for index, sentence in enumerate(document):
        # Extract named entities from the sentence
        local_relationships = {}
        ner_results = ner_pipeline(sentence)

        # Extract entities from NER results
        entities = [{"text": ent["word"], "label": ent["entity_group"]} for ent in ner_results[0]]

        # Generate entity pairs
        entity_pairs = []
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i < j:  # Avoid duplicate pairs and self-pairing
                    entity_pairs.append((entity1, entity2))

        # Process each entity pair
        for pair_id, (entity1, entity2) in enumerate(entity_pairs):
            # Format the sentence for the relationship extraction model
            input_text = f"Extract relations between entities in the text: <e1>{entity1['text']}</e1> is related to <e2>{entity2['text']}</e2>."

            # Tokenize and encode the input
            inputs = tokenizer_for_relationship(input_text, return_tensors="pt", padding=True, truncation=True)

            # Predict the relationship
            with torch.no_grad():  # Disable gradient computation
                outputs = model_for_relationship(**inputs)

            # Get the predicted relation label
            logits = outputs.logits
            predicted_label_id = logits.argmax(dim=-1).item()
            predicted_relation = model_for_relationship.config.id2label[predicted_label_id]

            # Add the relationship to the local dictionary
            relationship_id = f"rel_{index}_{pair_id}"
            local_relationships[relationship_id] = {
                "entity1": entity1["text"],
                "entity2": entity2["text"],
                "relationship": predicted_relation,
                "sentence_index": index
            }

        # Append the local relationships to the global dictionary
        relationships.update(local_relationships)

    # Store all relationships in a single JSON file
    output = {"relationships": relationships}

    with open("extracted_relationships.json", "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print(f"Relationships extracted and stored in 'extracted_relationships.json'")



# os.environ["GEMINI_API_KEY"] = "AIzaSyAJb1jtEHzQEeU6elwUp3iXXwt1edwwQu0"
# 
# 
# # Load the Flair NER tagger
# tagger = SequenceTagger.load('de-ner-large')
# 
# extract_knowledge(0, [], tagger)

    if DEBUG:
        print(f"Document {doc_index} processed.")
        print(f"Extracted entities: {len(global_entities_dict)}")
        print(f"Relationships (not implemented yet): {len(relationships)}")

    return global_entities_dict, relationships    


# os.environ["GEMINI_API_KEY"] = "AIzaSyAJb1jtEHzQEeU6elwUp3iXXwt1edwwQu0"
# 
# 
# # Load the Flair NER tagger
# tagger = SequenceTagger.load('de-ner-large')
# 
# extract_knowledge(0, [], tagger)
