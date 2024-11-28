from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from torch.optim import AdamW

"""
input parameter 
"tagger" : the model which you want to fine tune
"data_folder" : directory where files are located in
"train file" : .txt file which will be used for training while fine tuning
"dev_file" : .txt file which will be used for validating while fine tuning
"test_feil" : .txt file which will be used for testing while fine tuning

This function make the fine tuned model in the directory.
If fine-tuning is done, the model will be saved as './fine_tuned_ner/final-model.pt'
We can use this model to extract named entities.

sentence = Sentence("Examples")
fine_tuned_tagger.predict(sentence)
"""
def flair_finetuning(tagger,data_folder,train_file,dev_file,test_file):
    # 1. Define the columns in your dataset
    columns = {0: 'text', 1: 'ner'}  # 0th column is token, 1st column is NER tag

    # 2. Load your dataset
    corpus = ColumnCorpus(data_folder, columns,
                        train_file = train_file,
                        dev_file = dev_file,
                        test_file = test_file)

    # 3. Prepare ModelTrainer
    trainer = ModelTrainer(tagger, corpus)

    # 4. Start fine-tuning
    trainer.fine_tune(
        './fine_tuned_flair',  # Output directory
        learning_rate=3e-5,
        mini_batch_size=16,
        max_epochs=10,  # Number of epochs
        optimizer=AdamW  # Optimizer
    )
