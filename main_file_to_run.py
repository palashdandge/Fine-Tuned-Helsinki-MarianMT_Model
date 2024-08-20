!pip install kaggle
!kaggle datasets download -d polly42rose/indian-accent-dataset


!pip install transformers datasets torch
!pip install sentencepiece


import zipfile

# Define the path to the ZIP file
zip_file_path = 'indian-accent-dataset.zip'
extract_folder_path = '/content/indian-accent-dataset'

# Create the directory if it doesn't exist
import os
os.makedirs(extract_folder_path, exist_ok=True)

# Extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder_path)

print("Extraction complete!")


import os

# List files in the extracted folder
for root, dirs, files in os.walk(extract_folder_path):
    for file in files:
        print(os.path.join(root, file))






import os
import pandas as pd
from datasets import Dataset
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments



# Define the path to your train.txt file
train_txt_path = '/content/indian-accent-dataset/metadata/metadata/train.txt'  # Update if necessary

# Function to create the dataset
# Function to create the dataset
def create_dataset(train_txt_path):
    data = []

    with open(train_txt_path, 'r') as file:
        for line in file:
            # Assuming the format is: source_text|target_text|other_values...
            values = line.strip().split('|')
            
            # Adjust this based on the actual format of your file
            if len(values) >= 2:
                source_text = values[0]
                target_text = values[1]

                data.append({
                    'source_text': source_text,
                    'target_text': target_text
                })
            else:
                print(f"Skipping line due to incorrect format: {line.strip()}")

    return Dataset.from_pandas(pd.DataFrame(data))


# Create the dataset
dataset = create_dataset(train_txt_path)
def train_model(dataset):
    model_name = "Helsinki-NLP/opus-mt-en-de"  # Model for English-to-German translation
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Preprocess function to tokenize the dataset
    def preprocess_function(examples):
        inputs = tokenizer(examples['source_text'], max_length=128, truncation=True, padding='max_length')
        targets = tokenizer(examples['target_text'], max_length=128, truncation=True, padding='max_length')

        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': targets['input_ids']
        }
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        remove_unused_columns=False,
    )

    # Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,  # Normally you'd use a separate validation dataset
    )

    trainer.train()
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

# Train the model
train_model(dataset)




########################## Inference or Testing ##############################


LANGUAGES = {
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'mr':"Marathi",
    'hi':'Hindi'
    # Add other languages as needed
}
from transformers import MarianMTModel, MarianTokenizer

def translate(text, model, tokenizer, source_lang, target_lang):
    # Prepare inputs for translation
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

def load_model_and_tokenizer(source_lang, target_lang):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

# Define your source and target languages
source_lang = 'en'  # Example: English
target_lang = 'hi'  # Example: French

# Load the appropriate model and tokenizer
model, tokenizer = load_model_and_tokenizer(source_lang, target_lang)

# Translate new sentences
inputs = [
    "This is a test sentence.",
    "How are you today?",
    "Let's see how well this model performs."
]

for input_text in inputs:
    translation = translate(input_text, model, tokenizer, source_lang, target_lang)
    print(f"Input: {input_text}")
    print(f"Translation: {translation}")