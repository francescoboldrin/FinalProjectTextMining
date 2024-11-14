"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-11-13 11:19:54
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-11-13 11:22:21
FilePath: scripts/Evaluator.py
Description: the file contains the functions to evaluate the algorithm developed, yet to decide how
"""

import DataLoader
import json
from collections import defaultdict

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

results = ner_evaluate('dataset_gt/test_NER_gt.json', 'NER_extracted_file_example/GPT_generated_example.json')
print(json.dumps(results, indent=4))