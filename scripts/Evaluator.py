"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-11-13 11:19:54
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-11-28 19:18:50
FilePath: scripts/Evaluator.py
Description: the file contains the functions to evaluate the algorithm developed, yet to decide how
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import DataLoader
import json
from collections import defaultdict
from sklearn.metrics import jaccard_score, confusion_matrix
import numpy as np

from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

# # Load the pre-trained BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")


def distance_bert(str1, str2):
    """
    Calculate the cosine distance between two strings. using BERT embeddings.
    
    Parameters:
    - str1 (str): The first string.
    - str2 (str): The second string.
    
    Returns:
    - float: 1 - cosine similarity between the two strings.
    """
    # Encode the strings using the BERT tokenizer
    inputs = tokenizer([str1, str2], return_tensors="pt", padding=True, truncation=True)
    
    # Get the embeddings from the BERT model
    with torch.no_grad():
        output = model(**inputs)
    
    # Extract the embeddings for the [CLS] token
    embeddings = output.last_hidden_state[:, 0, :]
    
    # Calculate the cosine similarity
    similarity = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    
    # Return the cosine distance
    return 1 - similarity.item()

def distance_jaccard(str1, str2):
    """
    Calculate the Jaccard distance between two strings.
    
    Parameters:
    - str1 (str): The first string.
    - str2 (str): The second string.
    
    Returns:
    - float: The Jaccard distance between the two strings.
    """
    # Tokenize the strings
    tokens1 = set(str1.split())
    tokens2 = set(str2.split())
    
    # implement the Jaccard distance
    set_intersection = len(tokens1.intersection(tokens2))
    set_union = len(tokens1.union(tokens2))
    
    jaccard_distance = 1 - (set_intersection / set_union) if set_union > 0 else 0
    
    return jaccard_distance

def ner_evaluate(gt_file_path:str, rec_file_path:str):
    
    # 1) load groundtruth
    ner_gt = DataLoader.read_linked_docred(gt_file_path)

    # 2) load recognized entity
    ner_rec = DataLoader.read_linked_docred(rec_file_path)

    # 3) evaluate
    """
    Evaluate for each documents,
    : Calculate metrics by comparing the type of the TOKENS one by one
    --------------------------------
    But I'm not sure yet if it also works if entity has two or more token.
    ex) entity : Albert Einstein (PER)
        token : Albert (B-PER) / Einstein (I-PER)
    """
    
    #for i in range(len(ner_gt)):
    #    gt_doc = ner_gt[i]
    #    rec_doc = ner_rec[i]
    final_metrics = []
    for i in range(0,1):
        gt_doc = ner_gt[2]
        rec_doc = ner_rec[0]

        results = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
        total_entities = 0

        # add TP, FP, FN by searching whole sentences
        for gt_sent, rec_sent in zip(gt_doc['sents'], rec_doc['sents']):
            gt_entities = {(entity['name'], entity['type'].replace("B-", "").replace("I-", "")) for entity in gt_sent['entities']}
            rec_entities = {(entity['name'], entity['type']) for entity in rec_sent['entities']}

            for entity in rec_entities & gt_entities:
                results[entity[1]]['TP'] += 1  # store for each entity type
        
            for entity in rec_entities - gt_entities:
                results[entity[1]]['FP'] += 1
            
            for entity in gt_entities - rec_entities:
                results[entity[1]]['FN'] += 1

            total_entities += len(gt_entities)

        metrics = {}
        total_TP = total_FP = total_FN = 0

        # evaluation for each entity type
        for entity_type, counts in results.items():
            TP, FP, FN = counts['TP'], counts['FP'], counts['FN']
            total_TP += TP
            total_FP += FP
            total_FN += FN

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = TP / total_entities if total_entities > 0 else 0

            metrics[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy
            }

        # evaluation for overall
        total_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        total_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        total_f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
        overall_accuracy = total_TP / total_entities if total_entities > 0 else 0

        metrics['overall'] = {
            'precision': total_precision,
            'recall': total_recall,
            'f1_score': total_f1_score,
            'accuracy': overall_accuracy
        }

        final_metrics.append({'title':gt_doc.get('title', ''),'metric':metrics})
    
    return final_metrics

# results = ner_evaluate('dataset_gt/test_NER_gt.json', 'NER_extracted_file_example/GPT_generated_example.json')
# print(json.dumps(results, indent=4))


def extract_names_and_labels_from_gt(dataset, num_documents):
    """
    Extracts the 'name' and 'type' (label) of entities from the first `num_documents` in the dataset.

    :param dataset: List of documents, where each document is a dictionary with keys:
                    'title', 'sents', 'entities', 'relations'.
    :param num_documents: Number of documents to process (starting from the first document).
    :return: Dictionary with document indices as keys and lists of extracted entity names and types as values.
             Each entity is represented by a dictionary containing 'name' and 'label' (type).
    """
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list of documents.")
    if num_documents <= 0:
        return {}

    extracted_entities = {}

    for doc_index, document in enumerate(dataset[:num_documents]):
        if 'entities' not in document:
            raise ValueError(f"Document at index {doc_index} is missing the 'entities' key.")

        entities_list = []

        for entity in document['entities']:
            # Extract relevant details for each entity (only name and type)
            entity_details = {
                # Get all mentions of the entity and extract the names
                "name": [entity.get("mentions", [{}])[i].get("name", "") for i in range(len(entity.get("mentions", [{}])))],
                
                "label": entity.get("type", "")
            }
            entities_list.append(entity_details)

        # Store extracted name and label for the current document
        extracted_entities[doc_index] = entities_list

    return extracted_entities


def extract_names_and_labels_from_dataset(dataset, num_documents):
    """
    Extracts the 'name' and 'label' (type) of entities from the first `num_documents` in the dataset,
    where each document is indexed by a number and contains a dictionary of entities.

    :param dataset: Dictionary where the key is the document index, and the value is another dictionary containing
                    entity names as keys and a list with the type and indices as values.
    :param num_documents: Number of documents to process (starting from the first document).
    :return: Dictionary with document indices as keys and lists of extracted entity names and types as values.
             Each entity is represented by a dictionary containing 'name' and 'label' (type).
    """
    if not isinstance(dataset, dict):
        raise ValueError("Dataset must be a dictionary of documents.")
    if num_documents <= 0:
        return {}

    extracted_entities = {}

    for doc_index in range(min(num_documents, len(dataset))):
        document = dataset[doc_index]

        # List to store the entities for the current document
        entities_list = []

        # Loop through the entities in the document
        for entity_name, (entity_label, _) in document.items():
            # Create a dictionary with the name and label for each entity
            entity_details = {
                "name": entity_name,
                "label": entity_label
            }
            entities_list.append(entity_details)

        # Store the extracted entity names and labels for the current document
        extracted_entities[doc_index] = entities_list

    return extracted_entities


def plot_confusion_matrix(confusion_df):
    """
    Plots a heatmap for the given confusion matrix DataFrame.

    Parameters:
    confusion_df (pd.DataFrame): The confusion matrix as a pandas DataFrame with labels as index and columns.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_df, annot=True, fmt=".2f", cmap="Blues", cbar=True, linewidths=0.5)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def check_similarity(ent_retr_name, list_ent_name):
    for name in list_ent_name:
        if distance_jaccard(ent_retr_name, name) < 0.2:
            return True
    return False

"""
Even if it's same entity, there are several different names
ex. Zest Airways, Inc. & Zest Air
To avoid this, I made gt file, which 'head' and 'tail' is not a single word,
but the list of all same entity words

usage example :
rel_gt = read_linked_docred("./dataset_Linked-DocRED/train_annotated.json")
gt_relations = extract_relations_from_gt(rel_gt, 10)

output : 
list of docs
each docs have list of relations
each relations have "head", "type", "tail",
"head" and "tail" have the list of same entity words
"""
def extract_relations_from_gt(dataset, num_documents):

    relation_dict = DataLoader.read_linked_docred('./dataset_Linked-DocRED/rel_info.json')

    docs_list = []

    for doc_index, document in enumerate(dataset[:num_documents]):
        extracted_doc = {}
        extracted_doc["doc_index"] = doc_index

        relation_list = []

        entities = document['entities']
        for relation in document['relations']:
            head_entity = relation.get("h", "")
            tail_entity = relation.get("t", "")
            entity_details = {
                "head": list(set([mention.get("name", "")for mention in entities[head_entity]["mentions"]])),
                "type": relation_dict[relation.get("r", "")],
                "tail": list(set([mention.get("name", "")for mention in entities[tail_entity]["mentions"]])),
            }
            relation_list.append(entity_details)

        # Store extracted name and label for the current document
        extracted_doc["relationships"] = relation_list
        docs_list.append(extracted_doc)

    return docs_list


def evaluate_ner(gt_file_path, extracted_file_path):
    """
        This function evaluates the performance of an entity extraction system by comparing the ground truth entities 
        with the extracted entities. It calculates two metrics:

        1. **Jaccard Distance**: A measure of similarity between the ground truth and extracted entity sets.
        2. **Confusion Matrix**: A table to compare the predicted labels with the ground truth labels, specifically 
           for the relevant entity types.

        The function performs the following steps:

        1. Loads the ground truth entities and extracted entities.
        2. Compares each entity from the ground truth to the extracted entities based on their names and labels.
        3. Calculates the **Jaccard Distance** between ground truth and extracted entities for each document.
        4. Builds a **Confusion Matrix** to track the performance of the extraction system across the relevant entity 
           labels (`MISC`, `ORG`, `PER`, `LOC`, `NUM`).
        5. Prints out some debug information for manual inspection.

        Parameters:
        - **gt_file_path** (`str`): The file path to the JSON file containing the ground truth entity annotations. 
          The expected format is a list of documents, each containing named entities with labels.

        - **extracted_file_path** (`str`): The file path to the JSON file containing the extracted entities. 
          Similar to the ground truth file, it should be structured to map documents to entities with labels.

        Returns:
        - **confusion_df** (`pd.DataFrame`): A Pandas DataFrame representing the confusion matrix. The rows and columns 
          represent the ground truth and extracted labels, respectively. Each cell contains the count of instances for the 
          corresponding pair of labels.

        Functionality:
        1. **Loading Data**:  
           - Ground truth data is loaded using the `DataLoader.read_linked_docred(gt_file_path)` function.
           - Extracted data is loaded using `DataLoader.read_extracted_entities_json(extracted_file_path)`.

        2. **Entity Comparison**:  
           - For each document, the entities are compared between the ground truth and the extracted entities. 
             If the entity names and labels match, it's counted as a correct prediction. 
             If only the entity names match but the labels differ, it's a misclassification.

        3. **Jaccard Distance**:  
           - For each document, the Jaccard score between the set of ground truth entities and the set of extracted 
             entities is calculated. The Jaccard score is defined as:
             \[
             Jaccard\_score = 1 - \frac{|Intersection|}{|Union|}
             \]
             where `Intersection` and `Union` refer to the sets of entity names in the ground truth and extracted 
             entities, respectively. If there are no entities in the union, the score is set to 1.0 (indicating maximum distance).

        4. **Confusion Matrix**:  
           - A confusion matrix is created to compare the ground truth labels with the extracted labels. This matrix is 
             initialized to zeros and updated based on the comparisons of labels. The relevant labels tracked are 
             `['MISC', 'ORG', 'PER', 'LOC', 'NUM']`.

        5. **Debug Output**:  
           - The function prints out a comparison between the ground truth and extracted entities for each document for 
             manual verification.
           - After the evaluation, it prints statistics such as the total number of ground truth entities and the number 
             of failures (misclassifications).

        Example Usage:
        ```python
        # Evaluate the performance of the entity extraction system
        confusion_matrix_df = evaluate('./dataset_Linked-DocRED/train_annotated.json', './extracted_entities.json')

        # Print out the confusion matrix
        print(confusion_matrix_df)
        ```

        Notes:
        - This function prints intermediate results for debugging and inspection, such as ground truth and extracted 
          entities for each document, as well as the confusion matrix summary (total ground truth entities and failures).
        - The current version does not compute the Jaccard Distance scores in the output but calculates them internally. 
          You can add further analysis or store these scores as needed.
        - The `evaluate` function assumes that the entities are structured with at least the fields `name` and `label`.
        """
    # TODO: revise the evaluate function
    

    # Define the relevant labels
    relevant_labels = ['MISC', 'ORG', 'PER', 'LOC']

    # 1) load groundtruth
    ner_gt = DataLoader.read_linked_docred(gt_file_path)

    gt_entities = extract_names_and_labels_from_gt(ner_gt, 10)

    ner_ev = DataLoader.read_extracted_entities_json(extracted_file_path)

    extracted_entities = extract_names_and_labels_from_dataset(ner_ev, 10)
    
    for doc in gt_entities:
        print("\nGROUNDTRUTH ENTITIES:")
        for entiti in gt_entities[doc]:
            if entiti['label'] in relevant_labels:
                print(entiti)
        print("\n--\n")
        print("RETRIEVED ENTITIES:")
        for entiti2 in extracted_entities[doc]:
            if entiti2['label'] in relevant_labels:
                print(entiti2)
        print("\n\n\n--------------\n\n")

    # Define the labels to track

    # Initialize the confusion matrix with zeros for each label pair
    # Assuming `relevant_labels`, `gt_entities`, and `extracted_entities` are provided as input.

    # Create a confusion matrix initialized with zeros
    confusion_matrix = [[0.0 for _ in range(len(relevant_labels))] for _ in range(len(relevant_labels))]

    # Create a label-to-index map for easy access
    label_to_idx = {label: idx for idx, label in enumerate(relevant_labels)}

    counter_failure = 0
    tot_gt_entities = 0
    finded_ent = 0

    # Compare ground truth and extracted entities
    for doc_index in gt_entities:
        if doc_index in extracted_entities:
            # Loop over all ground truth entities in the document
            for gt_entity in gt_entities[doc_index]:
                gt_label = gt_entity['label']

                # 1) Check if the label is a relevant label
                if gt_label in relevant_labels:
                    tot_gt_entities += 1

                    # Flag to track if this entity was matched
                    entity_matched = False
                    
                    print("searching for: ", gt_entity)

                    # Loop over the extracted entities for this document
                    for ex_ent in extracted_entities[doc_index]:
                        ex_label = ex_ent['label']

                        # 3) Compare the entity name and the labels
                        if check_similarity(ex_ent['name'], gt_entity['name']):  # Match by name
                            try:
                                gt_idx = label_to_idx[gt_label]
                                ex_idx = label_to_idx[ex_label]
                                
                                print("found: ", ex_ent)
                                print("\n\n")
    
                                # Increment the corresponding position in the confusion matrix
                                confusion_matrix[gt_idx][ex_idx] += 1
                                finded_ent += 1
                                entity_matched = True
                            except KeyError:
                                continue
                            
                            # erase the entity from the list of the extracted entities
                            extracted_entities[doc_index].remove(ex_ent)
                            break

                    # If no match was found, count it as a failure
                    if not entity_matched:
                        counter_failure += 1
                        
    confusion_matrix_normalized = confusion_matrix
    
    # find F1 score
    f1_score = 0
    for i in range(len(confusion_matrix)):
        tp = confusion_matrix[i][i]
        fp = sum([confusion_matrix[j][i] for j in range(len(confusion_matrix))]) - tp
        fn = sum(confusion_matrix[i]) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score += 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    f1_score /= len(confusion_matrix)

    # Normalize the confusion matrix to values between 0 and 1
    # Normalization is done row-wise, dividing by the total entities for that ground truth label
    for i, row in enumerate(confusion_matrix):
        row_sum = sum(row)
        if row_sum > 0:  # To avoid division by zero
            confusion_matrix[i] = [value / row_sum for value in row]

    # Output results for verification
    print("Confusion Matrix (Normalized):")
    for row in confusion_matrix:
        print(row)

    total_ground_truth_entities = tot_gt_entities  # Total entities in ground truth
    total_found_entities = finded_ent  # Total entities correctly identified
    total_failures = counter_failure  # Total entities missed

    # Calculate success percentage
    success_percentage = (total_found_entities / total_ground_truth_entities) * 100 if total_ground_truth_entities > 0 else 0

    # Print results
    print(f"Total Ground Truth Entities: {total_ground_truth_entities}")
    print(f"Total Found Entities: {total_found_entities}")
    print(f"Total Failures: {total_failures}")
    print(f"Success Rate: {success_percentage:.2f}%")
    print(f"F1 Score (average between the labels): {f1_score:.2f}")

    # Convert the confusion matrix to a DataFrame for better readability
    
    confusion_df = pd.DataFrame(confusion_matrix, index=relevant_labels, columns=relevant_labels)
    
    plot_confusion_matrix(confusion_df)
    
    return confusion_df
                            

# results = evaluate('./dataset_Linked-DocRED/train_annotated.json', "./extracted_entities_llm_gemini.json")
# 
# print(results)

def parse_relations_ex(ex, num_of_documents):
    """
    from: 0: [{'head': 'AirAsia Zest', 'type': 'airline hub', 'tail': 'Ninoy Aquino International Airport'}, {'head': 'AirAsia Zest', 'type': 'headquarters location', 'tail': 'Pasay City'}, {'head': 'AirAsia Zest', 'type': 'country', 'tail': 'Philippines'},
    to: 
        (head, type, tail)
    """
    relations_ev = []
    
    for i in range(num_of_documents):
        document = ex[i]
        for relation in document:
            head = relation['head']
            tail = relation['tail']
            relations_ev.append((head, relation['type'], tail))
            
    return relations_ev

def parse_relations_gt(gt, num_of_documents):
    # retrieve the relationships from the ground truth in the first `num_of_documents`
    relations_gt = []
    
    for i in range(num_of_documents):
        document = gt[i]
        for relation in document['relations']:
            head = relation['h']
            tail = relation['t']
            
            # the number in h and t is the index of the entity in the entities list
            # the relation is the relation between the two entities
            # find the actual entities in the entities list
            head_entity = document['entities'][head]
            tail_entity = document['entities'][tail]
            
            #find pnly the first mention of the entity
            head_entity = head_entity['mentions'][0]['name']
            tail_entity = tail_entity['mentions'][0]['name']
            
            relations_gt.append((head_entity, relation['r'], tail_entity))
            
    
    return relations_gt


def load_gt_relations(file_path):
    """
    Load the ground truth relationships from the specified file path.

    Parameters:
    - file_path (str): The file path to the JSON file containing the ground truth relationships.
    
    from this: 
    {
        "doc_index": 0,
        "relationships": [
            {
                "head": [
                    "AirAsia Zest",
                    "Asian Spirit",
                    "Zest Air",
                    "Zest Airways, Inc.",
                    "Asian Spirit and Zest Air"
                ],
                "type": "headquarters location",
                "tail": [
                    "Pasay City"
                ]
            },
    to this:
    [(head, type, tail), ...]
    """
    
    with open(file_path, "r") as file:
        data = json.load(file)
        
    relations_gt = []
    
    for doc in data:
        for relation in doc['relationships']:
            head = relation['head']
            tail = relation['tail']
            relations_gt.append((head, relation['type'], tail))
            
    return relations_gt


def evaluate_relationship_extraction(file_path_gt, file_path_ev): # file path to the ground truth and the extracted file
    """
    Evaluate the performance of a relationship extraction system by comparing the ground truth relationships with the
    extracted relationships. The function calculates the precision of the extracted relationships based on the ground
    truth relationships.
    
    Parameters:
    - file_path_gt (str): The file path to the JSON file containing the ground truth relationships.
    - file_path_ev (str): The file path to the JSON file containing the extracted relationships.
    
    Returns:
    - None: The function prints the found relations, ground truth relations not found, and extracted relations not found.
    
    Additional Notes:
    - The function uses the `parse_relations_gt` and `parse_relations_ex` functions to extract the relationships from
      the ground truth and extracted files, respectively.
    - The function calculates the precision of the extracted relationships by comparing the relationships based on the
        head entity, relation type, and tail entity.
    """
    
    gt = load_gt_relations(file_path_gt)
    ex = DataLoader.read_extracted_relations_json(file_path_ev)
    
    rel_exs = parse_relations_ex(ex, 10)
    
    for i in range(10):
        print(rel_exs[i])
    print("\n\n")
    for i in range(10):
        print(gt[i])
    
    print("\n\n")
    rel_gts = gt
    
    
    found_relations = []
    gt_not_found = []
    ex_not_found = []
    
    # build a confusion matrix with the labels (so the type field in the relationships)
    
    
    # print("Labels: ", labels)
    
    print("labels: ", set([rel[1] for rel in rel_gts]))
    
    
    for rel_gt in rel_gts:
        found = False
        
        print("Searching for: ", rel_gt)
        for rel_ex in rel_exs:
            # THIS IS THE IMPORTANT PART WE SEARCH FOR RELATIONSHIPS WITH THE SAME HEAD AND TAIL (OR AT LEAST THE SAME MEANING SUCH AS Zest AirAsia and AirAsia Zest)
            if check_similarity(rel_ex[0], rel_gt[0]) and check_similarity(rel_ex[2], rel_gt[2]):
                found = True
                print("Found: ", rel_ex)
                found_relations.append(rel_gt)
                break
        
        if not found:
            gt_not_found.append(rel_gt)

    labels = [l for l in set([rel[1] for rel in found_relations])]

    confusion_matrix = np.zeros((len(labels), len(labels)))

    # Create a label-to-index map for easy access
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    for rel_gt in found_relations:
        for rel_ex in rel_exs:
            if check_similarity(rel_ex[0], rel_gt[0]) and check_similarity(rel_ex[2], rel_gt[2]):
                try:
                    gt_idx = label_to_idx[rel_gt[1]]
                    ex_idx = label_to_idx[rel_ex[1]]
                    confusion_matrix[gt_idx][ex_idx] += 1
                except KeyError:
                    continue
    
    print("Found relations:")
    print(found_relations)
    
    print("Ground truth relations not found:")
    print(gt_not_found)
    
    # percentage of found relations
    precision = len(found_relations) / len(rel_gts)
    
    print("Precision: ", precision)
    
    
    # # Normalize the confusion matrix to values between 0 and 1
    # Normalization is done row-wise, dividing by the total entities for that ground truth label
    for i, row in enumerate(confusion_matrix):
        row_sum = sum(row)
        if row_sum > 0:  # To avoid division by zero
            confusion_matrix[i] = [value / row_sum for value in row]

    # transform the confusion matrix into a DataFrame
    confusion_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    
    plot_confusion_matrix(confusion_df)
    
    # print(relations_gt)
    # print(relations_ev)
    
    
# results = evaluate_relationship_extraction('./dataset_gt/train_annotated_relations.json', "./extracted_relationship_llm_gemini.json")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    