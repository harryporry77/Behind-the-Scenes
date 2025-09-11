import os
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from nnsight import NNsight
from transformers import WhisperForAudioClassification, WhisperFeatureExtractor
from peft import PeftModel
import matplotlib.pyplot as plt 
import seaborn as sns
import threading
model_lock = threading.Lock()

processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
id2label = {0: "anger", 1: "happiness", 2: "neutral", 3: "sadness"}
label2id = {v: k for k, v in id2label.items()}

def analyze_norms(nmodel, test_data, processor=processor, num_samples=10): 
    num_layers = nmodel.config.encoder_layers
    att_cos_all = 0
    mlp_cos_all = 0
    layer_cos_all = 0

    mean_relative_contribution_att = 0
    mean_relative_contribution_mlp = 0
    mean_relative_contribution_layer = 0

    total_sequence_length = 0  

    for i in tqdm(range(len(test_data.select(range(num_samples))))):
        sample = test_data[i]
        labels = torch.tensor([sample['labels']],device=nmodel.device)
        input_features = processor(
            sample['audio'],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(nmodel.device)
        
        att_cos = []
        mlp_cos = []
        layer_cos = []
        relative_contribution_att = []
        relative_contribution_mlp = []
        relative_contribution_layer = []
        
        with torch.no_grad():
            with nmodel.trace(input_features):
                seq_len = nmodel.encoder.layers[0].output[0].shape[1].save()
        
        total_sequence_length += seq_len
        
        with torch.no_grad():
            with nmodel.trace(input_features):
                for layer_idx in range(num_layers):
                    layer_inputs = nmodel.encoder.layers[layer_idx].input
                    self_attn_output = nmodel.encoder.layers[layer_idx].self_attn.output[0]
                    layer_outputs = nmodel.encoder.layers[layer_idx].output[0]
                    

                    attn_contribution = self_attn_output.detach()
                    mlp_input = (self_attn_output + layer_inputs).detach()
                    mlp_contribution = (layer_outputs - mlp_input).detach()

                    residual_norm = layer_inputs.detach().norm(dim=-1).float().clamp(min=1e-6)
                    
                    relative_contribution_att.append(
                        (attn_contribution.norm(dim=-1).float() / residual_norm).sum(1).cpu()
                    )
                    
                    relative_contribution_mlp.append(
                        (mlp_contribution.norm(dim=-1).float() / residual_norm).sum(1).cpu()
                    )
                    
                    total_layer_contribution = attn_contribution + mlp_contribution
                    relative_contribution_layer.append(
                        (total_layer_contribution.norm(dim=-1).float() / residual_norm).sum(1).cpu()
                    )
                    
                    att_cos.append(F.cosine_similarity(attn_contribution, layer_inputs.detach(), dim=-1).sum(1).cpu().float())
                    mlp_cos.append(F.cosine_similarity(mlp_contribution, layer_inputs.detach(), dim=-1).sum(1).cpu().float())
                    layer_cos.append(F.cosine_similarity(total_layer_contribution, layer_inputs.detach(), dim=-1).sum(1).cpu().float())
        
        mean_relative_contribution_att += torch.cat(relative_contribution_att, dim=0)
        mean_relative_contribution_mlp += torch.cat(relative_contribution_mlp, dim=0)
        mean_relative_contribution_layer += torch.cat(relative_contribution_layer, dim=0)
        
        att_cos_all += torch.cat(att_cos, dim=0)
        mlp_cos_all += torch.cat(mlp_cos, dim=0)
        layer_cos_all += torch.cat(layer_cos, dim=0)


    att_cos_all = att_cos_all / total_sequence_length
    mlp_cos_all = mlp_cos_all / total_sequence_length
    layer_cos_all = layer_cos_all / total_sequence_length

    mean_relative_contribution_att = mean_relative_contribution_att / total_sequence_length
    mean_relative_contribution_mlp = mean_relative_contribution_mlp / total_sequence_length
    mean_relative_contribution_layer = mean_relative_contribution_layer / total_sequence_length
    
    return (mean_relative_contribution_att, mean_relative_contribution_mlp, mean_relative_contribution_layer,
                att_cos_all, mlp_cos_all, layer_cos_all)
    

def logit_lens(nmodel, test_data):
    device = nmodel.device
    res_kl_divs = 0
    res_overlaps = 0
    cnt = 0
    
    all_layer_logits = []
    all_final_logits = []
    all_labels = []
    
    for sample in tqdm(test_data):
        kl_divs = []
        overlaps = []
        layer_logits = []  
        
        n_layers = nmodel.config.encoder_layers
        
        # Process input features
        input_features = processor(
            sample['audio'],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)
        labels = sample['labels']
        all_labels.append(labels)
        encoder_layers_outputs = []
        with torch.no_grad():
            with nmodel.trace(input_features):
                for l in range(n_layers):
                    encoder_layers_outputs.append(nmodel.encoder.layers[l].output[0].save())
                
                
                final_encoder_output = nmodel.encoder.output.last_hidden_state
                
                pooled_output = final_encoder_output.mean(dim=1)  
                projected = nmodel.projector(pooled_output)
                final_logits = nmodel.classifier(projected)
                
                all_final_logits.append(final_logits.save())
        

        for layer_idx, layer_output in enumerate(encoder_layers_outputs):
            pooled = layer_output.mean(dim=1)
            
            with torch.no_grad():
                projected = nmodel.projector(pooled)
                logits = nmodel.classifier(projected)
                layer_logits.append(logits)
        
        all_layer_logits.append(layer_logits)
        
        final_probs = F.softmax(final_logits, dim=-1)
        
        for layer_idx, layer_logit in enumerate(layer_logits):
            layer_probs = F.softmax(layer_logit, dim=-1)
            
            kl_div = F.kl_div(layer_probs.log(), final_probs, reduction='batchmean')
            kl_divs.append(kl_div.item())
            

            layer_pred = layer_logit.argmax(dim=-1)
            final_pred = final_logits.argmax(dim=-1)
            overlap = (layer_pred == final_pred).float().mean().item()
            overlaps.append(overlap)
        
        res_kl_divs += np.array(kl_divs)
        res_overlaps += np.array(overlaps)
        cnt += 1
    
    avg_kl_divs = res_kl_divs / cnt
    avg_overlaps = res_overlaps / cnt
    
    return {
        'avg_kl_divs': avg_kl_divs,
        'avg_overlaps': avg_overlaps,
        'all_layer_logits': all_layer_logits,
        'all_final_logits': all_final_logits,
        'all_labels': all_labels
    }
    

    
    
def residual_erasure(nmodel, test_data):

    device = next(nmodel.parameters()).device
    n_layers = nmodel.config.encoder_layers
    
    test_samples = test_data.shuffle().select(range(50))
    layer_contributions = []
    accuracy_drops = []
    loss_increases = []
    confidence_drops = []
    
    print("Calculating original model performance...")
    original_acc, original_loss, original_conf = evaluate_model(nmodel, test_samples)
    print(f"Original model - Accuracy: {original_acc:.3f}, Loss: {original_loss:.3f}, Confidence: {original_conf:.3f}")
    
    for layer_idx in tqdm(range(n_layers), desc="Residual Erasure Experiment"):
        
        erased_acc, erased_loss, erased_conf = evaluate_model_with_erasure(
            nmodel, test_samples, layer_idx
        )
        
        acc_drop = original_acc - erased_acc
        loss_increase = erased_loss - original_loss
        conf_drop = original_conf - erased_conf
        
        contribution = acc_drop + (conf_drop * 0.5) - (loss_increase * 0.1)
        
        layer_contributions.append(contribution)
        accuracy_drops.append(acc_drop)
        loss_increases.append(loss_increase)
        confidence_drops.append(conf_drop)
        
        print(f"Layer {layer_idx+1} - Accuracy Drop: {acc_drop:.3f}, Loss Increase: {loss_increase:.3f}, Confidence Drop: {conf_drop:.3f}")
        print(f"Layer {layer_idx+1} - Contribution Score: {contribution:.3f}")
    
    return {
        'layer_contributions': np.array(layer_contributions),
        'accuracy_drops': np.array(accuracy_drops),
        'loss_increases': np.array(loss_increases),
        'confidence_drops': np.array(confidence_drops),
        'original_metrics': (original_acc, original_loss, original_conf)
    }
    

def evaluate_model(nmodel, test_samples, batch_size=8):
    """Batch evaluation of model performance"""
    device = next(nmodel.parameters()).device
    correct = 0
    total_loss = 0
    total_confidence = 0
    total_samples = len(test_samples)
    
    # Process data in batches
    for i in tqdm(range(0, total_samples, batch_size), desc="Batch Model Evaluation"):
        batch_end = min(i + batch_size, total_samples)
        
        # Correct way: use select method to get batch data
        batch_indices = list(range(i, batch_end))
        batch_dataset = test_samples.select(batch_indices)
        
        # Process batch audio features
        batch_features = []
        batch_labels = []
        
        # Iterate through each sample in batch
        for idx in range(len(batch_dataset)):
            sample = batch_dataset[idx]  # Get single sample this way
            input_features = processor(
                sample['audio'],
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features
            batch_features.append(input_features.squeeze(0))
            batch_labels.append(sample['labels'])
        
        # Stack into batch tensors
        batch_features = torch.stack(batch_features).to(device)
        batch_labels = torch.tensor(batch_labels).to(device)
        
        with torch.no_grad():
            outputs = nmodel(batch_features)
            logits = outputs.logits
            
            # Calculate loss
            loss = F.cross_entropy(logits, batch_labels, reduction='sum')
            total_loss += loss.item()
            
            # Calculate accuracy
            pred = logits.argmax(dim=-1)
            correct += (pred == batch_labels).sum().item()
            
            # Calculate confidence
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1)[0].sum().item()
            total_confidence += confidence
    
    accuracy = correct / total_samples
    avg_loss = total_loss / total_samples
    avg_confidence = total_confidence / total_samples
    
    return accuracy, avg_loss, avg_confidence

def evaluate_model_with_erasure(nmodel, test_samples, erase_layer_idx, batch_size=4):
    """Batch evaluation of model performance with layer erasure"""
    device = next(nmodel.parameters()).device
    correct = 0
    total_loss = 0.0
    total_confidence = 0
    total_samples = len(test_samples)
    
    for i in tqdm(range(0, total_samples, batch_size), desc=f"Evaluating model (erasing layer {erase_layer_idx})"):
        batch_end = min(i + batch_size, total_samples)
        
        # Correct way: use select method to get batch data
        batch_indices = list(range(i, batch_end))
        batch_dataset = test_samples.select(batch_indices)
        
        # Prepare batch data
        batch_features = []
        batch_labels = []
        
        for idx in range(len(batch_dataset)):
            sample = batch_dataset[idx]
            input_features = processor(
                sample['audio'],
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features
            batch_features.append(input_features.squeeze(0))
            batch_labels.append(sample['labels'])
        
        batch_features = torch.stack(batch_features).to(device)
        batch_labels = torch.tensor(batch_labels).to(device)
        
        with torch.no_grad():
            with nmodel.trace() as tracer:
                with tracer.invoke(batch_features):
                    # Get input of layer to be erased
                    layer_input = nmodel.encoder.layers[erase_layer_idx].input.clone()
                    # Set layer output to its input (skip layer computation)
                    nmodel.encoder.layers[erase_layer_idx].output = (layer_input, None, None)
                    
                    # Get final encoder output
                    final_output = nmodel.encoder.output[0]
                    pooled_output = final_output.mean(dim=1)
                    projected = nmodel.projector(pooled_output)
                    logits = nmodel.classifier(projected).save()
        
        # Calculate metrics
        loss = F.cross_entropy(logits, batch_labels, reduction='sum')
        total_loss += loss.item()
        
        # Calculate accuracy
        pred = logits.argmax(dim=-1)
        correct += (pred == batch_labels).sum().item()
        
        # Calculate confidence
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0].sum().item()
        total_confidence += confidence
    
    accuracy = correct / total_samples
    avg_loss = total_loss / total_samples
    avg_confidence = total_confidence / total_samples
    
    return accuracy, avg_loss, avg_confidence


def progressive_residual_erasure(nmodel, test_data, num_samples=50):
    """
    Progressive Residual Erasure Experiment: Cumulatively remove multiple residual connections
    Starting from layer 0, gradually increase number of erased layers to observe cumulative effects
    """
    device = next(nmodel.parameters()).device
    n_layers = nmodel.config.encoder_layers
    
    # Get test samples
    test_samples = test_data.shuffle().select(range(min(num_samples, len(test_data))))
    
    # Store cumulative effects
    cumulative_results = []
    
    # Get original performance
    print("Calculating original model performance...")
    original_acc, original_loss, original_conf = evaluate_model(nmodel, test_samples)
    cumulative_results.append({
        'num_erased_layers': 0,
        'accuracy': original_acc,
        'loss': original_loss,
        'confidence': original_conf,
        'acc_drop': 0.0,
        'loss_increase': 0.0,
        'conf_drop': 0.0
    })
    
    print(f"Original model performance - Accuracy: {original_acc:.3f}, Loss: {original_loss:.3f}, Confidence: {original_conf:.3f}")
    
    # Gradually remove more layers
    for num_erased in tqdm(range(1, n_layers + 1), desc="Progressive Residual Erasure"):
        print(f"\nRemoving first {num_erased} layers' residual connections...")
        
        acc, loss, conf = evaluate_model_with_multiple_erasure(
            nmodel, test_samples, list(range(num_erased))
        )
        
        acc_drop = original_acc - acc
        loss_increase = loss - original_loss
        conf_drop = original_conf - conf
        
        cumulative_results.append({
            'num_erased_layers': num_erased,
            'accuracy': acc,
            'loss': loss,
            'confidence': conf,
            'acc_drop': acc_drop,
            'loss_increase': loss_increase,
            'conf_drop': conf_drop
        })
        
        print(f"After removing {num_erased} layers - Accuracy: {acc:.3f}, Loss: {loss:.3f}, Confidence: {conf:.3f}")
        print(f"Performance drops - Accuracy: {acc_drop:.3f}, Loss increase: {loss_increase:.3f}, Confidence drop: {conf_drop:.3f}")
    
    return cumulative_results

def evaluate_model_with_multiple_erasure(nmodel, test_samples, erase_layer_indices, batch_size=4):

    device = next(nmodel.parameters()).device
    correct = 0
    total_loss = 0
    total_confidence = 0
    total_samples = len(test_samples)
    
    for i in tqdm(range(0, total_samples, batch_size), desc=f"eval_model(erase_layer_indices={erase_layer_indices})"):
        batch_end = min(i + batch_size, total_samples)

        batch_indices = list(range(i, batch_end))
        batch_dataset = test_samples.select(batch_indices)

        batch_features = []
        batch_labels = []
        
        for idx in range(len(batch_dataset)):
            sample = batch_dataset[idx]
            input_features = processor(
                sample['audio'],
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features
            batch_features.append(input_features.squeeze(0))
            batch_labels.append(sample['labels'])
        
        batch_features = torch.stack(batch_features).to(device)
        batch_labels = torch.tensor(batch_labels).to(device)
        
        with torch.no_grad():
            with nmodel.trace() as tracer:
                with tracer.invoke(batch_features):
                    for layer_idx in erase_layer_indices:
                        layer_input = nmodel.encoder.layers[layer_idx].input.clone()
                        nmodel.encoder.layers[layer_idx].output = (layer_input, None, None)
                    
                    final_output = nmodel.encoder.output[0]
                    pooled_output = final_output.mean(dim=1)
                    projected = nmodel.projector(pooled_output)
                    logits = nmodel.classifier(projected).save()

        loss = F.cross_entropy(logits, batch_labels, reduction='sum')
        total_loss += loss.item()
        
        pred = logits.argmax(dim=-1)
        correct += (pred == batch_labels).sum().item()
        
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0].sum().item()
        total_confidence += confidence
    
    accuracy = correct / total_samples
    avg_loss = total_loss / total_samples
    avg_confidence = total_confidence / total_samples
    
    return accuracy, avg_loss, avg_confidence
