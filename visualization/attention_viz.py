import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import sys

def visualize_multiple_examples(model, tokenizer, sentences, device, figsize=(15, 4*5), output_file=None):
    """
    Visualize attention weights for multiple sentences.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer used to tokenize the input
        sentences: List of input sentences
        device: Device to run the model on
        figsize: Figure size for the plot
        output_file: Path to save the visualization (if None, just show the plot)
    """
    fig, axes = plt.subplots(len(sentences), 1, figsize=figsize)
    if len(sentences) == 1:
        axes = [axes]
    
    model.eval()

    for i, sentence in enumerate(sentences):
        # Tokenize the sentence
        token_ids = tokenizer.encode(sentence, add_special_tokens=True)
        input_ids = torch.tensor([token_ids]).to(device)

        with torch.no_grad():
            try:
                # Try with return_attention parameter
                outputs, attention_weights = model(input_ids, return_attention=True)
            except TypeError:
                print("Model doesn't support return_attention parameter. Make sure you've updated the model class.")
                sys.exit(1)

        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

        # Get class labels
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        
        # Extract attention weights
        attention = attention_weights.squeeze(0).cpu().numpy()  # [1, seq_len]

        all_tokens = []
        for id in token_ids:
            token = tokenizer.convert_ids_to_tokens(id)
            all_tokens.append(token)
        
        # Plot attention heatmap
        sns.heatmap(
            attention.reshape(1, -1),
            cmap="YlOrRd",
            annot=True,
            fmt=".3f",
            cbar=False,
            xticklabels=all_tokens,
            yticklabels=["Attention"],
            ax=axes[i]
        )
        
        # Rotate x-axis labels for better readability
        axes[i].set_xticklabels(all_tokens, rotation=45, ha="right", rotation_mode="anchor")
        axes[i].set_title(f"'{sentence}'\nPrediction: {sentiment} (Confidence: {confidence:.4f})")


    plt.tight_layout()
    
    # Save figure if output_file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {output_file}")
    
    # Show the plot
    plt.show()
    
    return fig

def analyze_attention_patterns(model, tokenizer, sentences, device):
    """
    Analyze attention patterns across multiple sentences.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer used to tokenize the input
        sentences: List of input sentences
        device: Device to run the model on
        
    Returns:
        Dictionary with analysis results
    """
    model.eval()
    results = []
    
    for sentence in sentences:
        # Tokenize the sentence
        token_ids = tokenizer.encode(sentence, add_special_tokens=True)
        input_ids = torch.tensor([token_ids]).to(device)
        tokens = [tokenizer.convert_ids_to_tokens(id) for id in token_ids]
        
        # Get model predictions and attention weights
        with torch.no_grad():
            outputs, attention_weights = model(input_ids, return_attention=True)
        
        # Get predicted class and probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
        
        # Extract attention weights
        attention = attention_weights.squeeze(0).cpu().numpy()
        
        # Ensure attention is in the right shape
        if len(attention.shape) == 1:
            # If it's a 1D array, reshape to 2D
            attention = attention.reshape(1, -1)
        
        # Find token with maximum attention
        if len(attention.shape) == 2:
            flat_idx = np.argmax(attention)
            # Convert flat index to 2D coordinates
            row_idx, col_idx = np.unravel_index(flat_idx, attention.shape)
            max_attention_idx = col_idx
            max_attention_value = attention[row_idx, col_idx]
        else:
            max_attention_idx = np.argmax(attention)
            max_attention_value = attention[max_attention_idx]
            
        max_attention_token = tokens[max_attention_idx]
        
        # Calculate mean and std of attention
        mean_attention = np.mean(attention)
        std_attention = np.std(attention)
        
        # Store results
        results.append({
            'sentence': sentence,
            'prediction': "Positive" if predicted_class == 1 else "Negative",
            'confidence': confidence,
            'tokens': tokens,
            'attention_values': attention.tolist(),
            'max_attention_token': max_attention_token,
            'max_attention_value': max_attention_value,
            'mean_attention': mean_attention,
            'std_attention': std_attention
        })
    
    return results