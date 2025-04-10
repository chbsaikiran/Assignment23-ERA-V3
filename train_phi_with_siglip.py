import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    SiglipVisionModel, 
    AutoTokenizer, 
    AutoImageProcessor, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from PIL import Image

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

class ImageTextProjection(nn.Module):
    def __init__(self, image_dim, text_dim):
        super().__init__()
        self.image_projection = nn.Linear(image_dim, text_dim)
        
    def forward(self, x):
        return self.image_projection(x)

def get_image_embedding(image, siglip_model, siglip_processor, linear_proj, device):
    with torch.no_grad():
        # Process image through SigLIP
        inputs = siglip_processor(image, return_tensors="pt")
        # Move inputs to the same device as model
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = siglip_model(**inputs)
        image_features = outputs.pooler_output
        
        # Project through trained linear layer
        projected_features = linear_proj(image_features)
        
    return projected_features

def main(
    num_images=100,
    batch_size=4,  # Smaller batch size due to memory constraints
    num_epochs=100,
    learning_rate=2e-4,
    questions=None  # List of 5 questions to be provided
):
    if questions is None or len(questions) != 5:
        print("Please provide exactly 5 questions!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SigLIP model and processor
    siglip_model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
    siglip_processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    
    # Load trained linear projection
    dummy_image = Image.new('RGB', (384, 384), color='black')
    with torch.no_grad():
        siglip_inputs = siglip_processor(dummy_image, return_tensors="pt")
        # Move inputs to device
        siglip_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in siglip_inputs.items()}
        siglip_outputs = siglip_model(**siglip_inputs)
        siglip_output_dim = siglip_outputs.pooler_output.shape[-1]
    
    # First load the checkpoint to get the correct output dimension
    checkpoint = torch.load('linear_projection_final.pth', map_location=device)
    output_dim = checkpoint['linear.weight'].shape[0]  # Get the output dimension from saved weights
    print(f"Loading linear projection with output dimension: {output_dim}")
    
    # Initialize linear projection with correct dimensions
    linear_proj = LinearProjection(siglip_output_dim, output_dim).to(device)
    try:
        linear_proj.load_state_dict(checkpoint)
        print("Successfully loaded linear projection weights")
    except Exception as e:
        print(f"Error loading linear projection weights: {e}")
        return

    # Load Phi model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    phi_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        quantization_config=bnb_config,
        device_map="auto"
    )
    phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    
    # Add padding token if not present
    if phi_tokenizer.pad_token is None:
        phi_tokenizer.pad_token = phi_tokenizer.eos_token
    
    # Get embedding dimension from phi model
    phi_embed_dim = phi_model.get_input_embeddings().weight.shape[1]
    
    # Create projection layer for image embeddings
    image_text_proj = ImageTextProjection(output_dim, phi_embed_dim).to(device)
    
    # Prepare model for k-bit training
    phi_model = prepare_model_for_kbit_training(phi_model)

    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["mlp.dense_h_to_4h", "mlp.dense_4h_to_h", "self_attn.qkv_proj", "self_attn.dense"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get PEFT model
    phi_model = get_peft_model(phi_model, lora_config)
    
    # Freeze SigLIP and linear projection
    for param in siglip_model.parameters():
        param.requires_grad = False
    for param in linear_proj.parameters():
        param.requires_grad = False

    # Load CIFAR10 test dataset
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])
    
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    subset_indices = list(range(num_images))
    subset_dataset = Subset(test_dataset, subset_indices)
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer for both phi model and image projection
    optimizer = AdamW([
        {'params': phi_model.parameters()},
        {'params': image_text_proj.parameters()}
    ], lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        phi_model.train()
        image_text_proj.train()
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            batch_size = images.size(0)

            # Get image embeddings
            image_embeddings = get_image_embedding(images, siglip_model, siglip_processor, linear_proj, device)

            # Process each question
            for q_idx, question in enumerate(questions):
                # Read corresponding answers
                answers = []
                for idx in range(batch_size):
                    global_idx = batch_idx * batch_size + idx
                    if global_idx < num_images:
                        file_path = f'qa_outputs/image_{global_idx}_extr.txt'
                        try:
                            with open(file_path, 'r') as f:
                                lines = f.readlines()
                                answer = lines[q_idx].strip() if q_idx < len(lines) else ""
                                answers.append(answer)
                        except:
                            answers.append("No answer available")

                # Tokenize questions and answers for the entire batch
                question_tokens = phi_tokenizer(
                    [question] * batch_size,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)

                target_tokens = phi_tokenizer(
                    answers,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)

                # Get question embeddings for the entire batch
                question_embeds = phi_model.get_input_embeddings()(question_tokens['input_ids'])  # [batch_size, seq_len, embed_dim]

                # Project and prepare image embeddings for the entire batch
                image_embeds = image_text_proj(image_embeddings)  # [batch_size, embed_dim]
                image_embeds = image_embeds.unsqueeze(1)  # [batch_size, 1, embed_dim]

                # Combine image embeddings with question embeddings
                combined_embedding = torch.cat([
                    image_embeds,  # [batch_size, 1, embed_dim]
                    question_embeds  # [batch_size, seq_len, embed_dim]
                ], dim=1)  # [batch_size, 1+seq_len, embed_dim]

                # Create attention mask for the combined sequence
                attention_mask = torch.ones(
                    (batch_size, combined_embedding.size(1)),
                    dtype=torch.long,
                    device=device
                )

                # Prepare labels by shifting them right
                labels = target_tokens['input_ids'].clone()
                labels = torch.cat([
                    torch.full((batch_size, combined_embedding.size(1) - 1), -100, device=device),
                    labels
                ], dim=1)[:, :combined_embedding.size(1)]

                # Forward pass
                outputs = phi_model(
                    inputs_embeds=combined_embedding,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                progress_bar.set_postfix({'loss': loss.item()})

        avg_epoch_loss = total_loss / (len(dataloader) * len(questions) * batch_size)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}')

    # Save the trained models
    phi_model.save_pretrained('phi_model_trained')
    torch.save(image_text_proj.state_dict(), 'image_text_proj.pth')
    print("Training completed. Models saved as 'phi_model_trained' and 'image_text_proj.pth'")

if __name__ == "__main__":
    # Example questions - replace with your actual questions
    questions = [
    "Give a description of the image?",
    "How does the main object in the image look like?",
    "How can the main object in the image be useful to humans?",
    "What is the color of the main object in the image?",
    "Describe the setting of the image?"
    ]
    
    main(questions=questions) 