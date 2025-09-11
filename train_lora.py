from transformers import WhisperForAudioClassification, WhisperFeatureExtractor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import recall_score
import evaluate
import numpy as np
import os

# ========== Label Mapping ==========
id2label = {0: "anger", 1: "happiness", 2: "neutral", 3: "sadness"}
label2id = {v: k for k, v in id2label.items()}
model_name = "openai/whisper-large-v2"

# ========== LoRA Config ==========
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["v_proj"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["projector", "classifier"],  
)

# ========== Processor ==========
processor = WhisperFeatureExtractor.from_pretrained(model_name)

# ========== Load & Preprocess Dataset ==========
ds = load_dataset("Zahra99/IEMOCAP_Audio")
dataset = concatenate_datasets([ds['session1'], ds['session2'], ds['session3'], ds['session4'], ds['session5']])
del ds 

def prepare_data(examples):
    audio_arrays = [item['array'] for item in examples['audio']]
    labels = examples['label']
    speakers = [path.split('/')[-1].split('_')[0] for path in [item['path'] for item in examples['audio']]]
    return {
        "audio": audio_arrays,
        "labels": labels,
        "speaker": speakers
    }

dataset = dataset.map(prepare_data, batched=True, remove_columns=dataset.column_names, num_proc=4)
speakers_array = np.array(dataset['speaker'])
all_speaker = sorted(list(set(dataset['speaker'])))

# ========== Data Collator ==========
def data_collator(features):
    batched_audio = []
    labels = []
    for feature in features:
        raw_audio = feature["audio"]
        label = feature["labels"]
        inputs = processor(raw_audio, sampling_rate=16000, return_tensors="pt")
        batched_audio.append(inputs.input_features[0])
        labels.append(label)
    # print("Input features shape:", torch.stack(batched_audio).shape)
    # print("Labels shape:", torch.tensor(labels).shape)
    return {
        "input_features": torch.stack(batched_audio),
        "labels": torch.tensor(labels)
    }

# ========== Metric ==========
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# ========== Cross-Validation ==========
results = []

for fold, test_speaker in enumerate(all_speaker):
    print(f"\n Fold {fold + 1}/10 | Test Speaker: {test_speaker}")
    
    test_indices = np.where(speakers_array == test_speaker)[0]
    train_indices = np.where(speakers_array != test_speaker)[0]
    
    test_data = dataset.select(test_indices)
    train_data = dataset.select(train_indices)


    model = WhisperForAudioClassification.from_pretrained(
        model_name,
        num_labels=4,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    print(f'load model :{model_name}')
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=f"./v_proj/{model_name}/fold{fold + 1}",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        num_train_epochs=20,
        logging_steps=10,
    
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
        dataloader_num_workers=4,
        label_names=["labels"],
        ddp_find_unused_parameters=False, 
        deepspeed="deepspeed.json",  
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    acc = eval_result["eval_accuracy"]
    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
    results.append(acc)

    from sklearn.metrics import confusion_matrix
    import json

    predictions = trainer.predict(test_data)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    cm = confusion_matrix(true_labels, pred_labels)

    cm_dict = {
        "confusion_matrix": cm.tolist(),
        "labels": list(id2label.values())
    }
    cm_path = f"./{model_name}-lora/fold{fold + 1}/confusion_matrix.json"
    with open(cm_path, 'w') as f:
        json.dump(cm_dict, f, indent=4)
    print(f" Confusion matrix saved to {cm_path}")

# ========== Final Results ==========
mean_acc = np.mean(results)
std_acc = np.std(results)
print(f"\nFinal 10-Fold Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
