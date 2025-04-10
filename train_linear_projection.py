import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import SiglipVisionModel, AutoTokenizer, AutoImageProcessor, AutoModel
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import argparse

def siglip_loss(image_embeddings, text_embeddings, temperature=0.07):
    # Normalize
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    # Compute pairwise similarities
    logits = image_embeddings @ text_embeddings.T  # [batch_size, batch_size]
    logits = logits / temperature

    # Ground truth: 1.0 for matching pairs (diagonal), 0.0 for all others
    batch_size = logits.size(0)
    targets = torch.eye(batch_size).to(logits.device)

    # Apply binary cross-entropy with logits
    loss = F.binary_cross_entropy_with_logits(logits, targets)

    return loss

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

def get_text_embedding(text, tokenizer, device, max_length=128):
    # Ensure text is not empty and has minimum content
    if not text or len(text.strip()) == 0:
        text = "This is a placeholder description."
    
    # Tokenize with padding and truncation
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding='max_length',  # Changed to max_length padding
        truncation=True,
        max_length=max_length  # Fixed max length for all inputs
    )
    
    # Move inputs to device and ensure correct data type
    inputs = {
        k: v.to(device).float() for k, v in inputs.items()
    }
    
    # Return the input_ids as embeddings
    return inputs['input_ids'].float()  # Convert to float for the loss calculation

def main(num_images=100, batch_size=32, num_epochs=50, learning_rate=1e-4, load_checkpoint=True, checkpoint_path='linear_projection.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models and processors
    siglip_model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
    siglip_processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Freeze SigLIP model
    for param in siglip_model.parameters():
        param.requires_grad = False
    
    siglip_model.to(device)

    # Get SigLIP output dimension and text embedding dimension
    # Create a proper dummy image (black image)
    dummy_image = Image.new('RGB', (384, 384), color='black')
    with torch.no_grad():
        siglip_inputs = siglip_processor(dummy_image, return_tensors="pt").to(device)
        siglip_outputs = siglip_model(**siglip_inputs)
        siglip_output_dim = siglip_outputs.pooler_output.shape[-1]
    
    # Get a sample text to determine embedding dimension
    dummy_text = "This is a test."
    dummy_embedding = get_text_embedding(dummy_text, tokenizer, device)
    text_embedding_dim = dummy_embedding.shape[-1]

    print(f"SigLIP output dimension: {siglip_output_dim}")
    print(f"Text embedding dimension: {text_embedding_dim}")

    # Create linear projection layer
    linear_proj = LinearProjection(siglip_output_dim, text_embedding_dim).to(device)

    # Load checkpoint if requested
    if load_checkpoint:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            linear_proj.load_state_dict(checkpoint)
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch instead.")
    
    # Load CIFAR10 test dataset
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])
    
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    subset_indices = list(range(num_images))
    subset_dataset = Subset(test_dataset, subset_indices)
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

    # Create text files directory if it doesn't exist
    os.makedirs('qa_outputs', exist_ok=True)

    # Optimizer
    optimizer = AdamW(linear_proj.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        linear_proj.train()
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            batch_size = images.size(0)

            # Get image embeddings
            with torch.no_grad():
                siglip_inputs = siglip_processor(images, return_tensors="pt").to(device)
                siglip_outputs = siglip_model(**siglip_inputs)
                image_features = siglip_outputs.pooler_output

            # Project image features
            projected_image_features = linear_proj(image_features)

            # Process text for each line (1 to 5)
            total_batch_loss = 0
            for line_num in range(5):
                text_embeddings_list = []
                
                # Read text from files for current batch
                for idx in range(batch_size):
                    global_idx = batch_idx * batch_size + idx
                    if global_idx < num_images:
                        file_path = f'qa_outputs/image_{global_idx}_extr.txt'
                        try:
                            with open(file_path, 'r') as f:
                                lines = f.readlines()
                                text = lines[line_num].strip() if line_num < len(lines) else ""
                        except:
                            text = "No description available"

                        # Get text embeddings directly from tokenizer
                        text_embedding = get_text_embedding(text, tokenizer, device)
                        text_embeddings_list.append(text_embedding)

                if text_embeddings_list:
                    # Stack instead of cat since all embeddings have same size now
                    text_embeddings = torch.stack(text_embeddings_list, dim=0).squeeze(1)
                    loss = siglip_loss(projected_image_features, text_embeddings)
                    total_batch_loss += loss

            # Average loss over all text lines
            avg_batch_loss = total_batch_loss / 5
            
            # Backpropagation
            optimizer.zero_grad()
            avg_batch_loss.backward()
            optimizer.step()

            total_loss += avg_batch_loss.item()
            progress_bar.set_postfix({'loss': avg_batch_loss.item()})

        avg_epoch_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}')

        # Save checkpoint after each epoch
        # checkpoint_dir = 'checkpoints'
        # os.makedirs(checkpoint_dir, exist_ok=True)
        # checkpoint_file = os.path.join(checkpoint_dir, f'linear_projection_epoch_{epoch+1}.pth')
        # torch.save(linear_proj.state_dict(), checkpoint_file)
        # print(f"Saved checkpoint to {checkpoint_file}")

    # Save final model
    torch.save(linear_proj.state_dict(), 'linear_projection_final.pth')
    print("Training completed. Final model saved as 'linear_projection_final.pth'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or continue training the linear projection layer')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to train on')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load_checkpoint', action='store_true', help='Whether to load from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='linear_projection.pth', help='Path to checkpoint file')
    
    args = parser.parse_args()
    main(
        num_images=args.num_images,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        load_checkpoint=args.load_checkpoint,
        checkpoint_path=args.checkpoint_path
    ) 