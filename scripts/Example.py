"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-11-06 14:59:51
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-11-06 16:02:58
FilePath: scripts/Example.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""

import spacy

# spacy.cli.download("en_core_web_sm")
# 
# nlp = spacy.load("en_core_web_sm")

# define the name of the entity
ner_categories = ["PERSON", "PRODUCT", "ORG"]

# define the text
example_text = (
    "Apple Inc. is an American multinational technology company headquartered in Cupertino, California, that was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976."
    "It designs, manufactures, and sells consumer electronics, computer software, and online services. It is considered one of the Big Tech technology companies, alongside Amazon, Google, Microsoft, and Facebook."
    "The company's hardware products include the iPhone smartphone, the iPad tablet computer, the Mac personal computer, the iPod portable media player, the Apple Watch smartwatch, the Apple TV digital media player, the AirPods wireless earbuds, the AirPods Max headphones, and the HomePod smart speaker."
    "Apple's software includes macOS, iOS, iPadOS, watchOS, and tvOS operating systems, the iTunes media player, the Safari web browser, the Shazam music identifier, and the iLife and iWork creativity and productivity suites, as well as professional applications like Final Cut Pro, Logic Pro, and Xcode."
)
# # create a spacy doc object
# doc = nlp(example_text)
# 
# # iterate over the entities
# for ent in doc.ents:
#     if ent.label_ in ner_categories:
#         print(ent.text, ent.label_)
#             

        
# Output: Apple ORG
# Output: Cupertino GPE
# Output: California GPE
# Output: Steve Jobs PERSON
# Output: Steve Wozniak PERSON
# Output: Ronald Wayne PERSON
# Output: 1976 DATE

# print the named entities

from flair.data import Sentence
from flair.nn import Classifier

# make a sentence
sentence = Sentence(example_text)

print(example_text)

# load the NER tagger
tagger = Classifier.load('de-ner-large')

print("Tagger loaded")

# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
# Access and print results
for entity in sentence.get_spans("ner"):
    print(entity)