import json

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


def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets


def store_triplets(index_document, triplets):
    # open the file extracted_relationship_big_with_rebel.json
    with open("../results/extracted_relationship_big_with_rebel.json", "r") as file:
        data = json.load(file)

    """
    store the data like this:
    {
        "doc_index": 0,
        "relationships": [
            {
                "head": "AirAsia Zest",
                "type": "airline hub",
                "tail": "Ninoy Aquino International Airport"
            },
            {
                "head": "AirAsia Zest",
                "type": "headquarters location",
                "tail": "Pasay City"
            },
            {
                "head": "AirAsia Zest",
                "type": "country",
                "tail": "Philippines"
            },
            {
                "head": "Ninoy Aquino International Airport",
                "type": "located in t....
    """
    # append the data to the relationships list
    data.append({
        "doc_index": index_document,
        "relationships": triplets
    })

    # write the data to the file
    with open("../results/extracted_relationship_big_with_rebel.json", "w") as file:
        json.dump(data, file)


def rebel_large_relation_extraction(index_document, document, triplet_extractor):
    """
    Extract relationships from the document using the REBEL model

    :param index_document: The index of the document
    :param document: The document to extract relationships from
    :param triplet_extractor: The REBEL model used to extract

    :return: None

    * The extracted relationships are stored in the file extracted_relationship_big_with_rebel.json
    """
    full_text = "".join([" ".join(sentence) for sentence in document])

    print("Extracting triplets from document", index_document)

    extracted_text = triplet_extractor.tokenizer.batch_decode([triplet_extractor(full_text, return_tensors=True, return_text=False)[0]["generated_token_ids"]])

    triplets = extract_triplets(extracted_text[0])

    print("Extracted", len(triplets), "triplets from document", index_document)

    store_triplets(index_document, triplets)