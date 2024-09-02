from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load the pre-trained LLaMA 3 model and tokenizer
model_name = "meta-llama/Llama-3"  # Replace with actual model path
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Load a translation dataset
dataset = load_dataset("wmt14", "de-en")  # German to English as an example

def tokenize_function(examples):
    return tokenizer(examples['de'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./llama3_translation")
tokenizer.save_pretrained("./llama3_translation")
