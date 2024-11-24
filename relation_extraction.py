from tqdm import tqdm
import spacy

def list_to_text(word_list):
    # Create a text string with proper formatting
    text = ""
    for i, word in enumerate(word_list):
        if word in [",", ".", ":", ")", "]", "!", "?"]:  # Don't add space before punctuation
            text += word
        elif i > 0 and word_list[i - 1] in ["(", "["]:  # Don't add space after opening brackets
            text += word
        else:
            text += " " + word if text else word  # Add space between words except the first one
    return text

nlp = spacy.load("en_core_web_sm")

def extract_entities_and_relationships(documents):
    all_entities = []
    all_triples = []
    
    # Use tqdm to wrap the outer loop for progress tracking
    for document in tqdm(documents, desc="Processing documents"):
        for sentences in tqdm(document, desc="Processing sentences", leave=False):  # Inner loop progress
            text = list_to_text(sentences)
            doc = nlp(text)
            
            triples = []  
            entities = []  

            for ent in doc.ents:
                entities.append((ent.text, ent.label_))  

            for sent in doc.sents:
                subject = None
                predicate = None
                obj = None

                for token in sent:
                    if token.dep_ == "nsubj":
                        subject = token.text
                    if token.pos_ == "VERB":
                        predicate = token.text
                    if token.dep_ == "dobj":
                        obj = token.text
                
                if subject and predicate and obj:
                    triples.append((subject, predicate, obj))
            
            all_entities.extend(entities)
            all_triples.extend(triples)
    
    return all_entities, all_triples