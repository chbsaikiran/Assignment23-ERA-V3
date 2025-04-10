import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

# Initialize model and processor
model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
    #_attn_implementation="flash_attention_2"
).to("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory
os.makedirs("SigLIP_Training/qa_outputs", exist_ok=True)

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage()
])

# Using test set instead of train set
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)

# List of questions
questions = [
    "Give a description of the image?",
    "How does the main object in the image look like?",
    "How can the main object in the image be useful to humans?",
    "What is the color of the main object in the image?",
    "Describe the setting of the image?"
]

def process_image(image, image_idx):
    # Create output file
    output_file = f"SigLIP_Training/qa_outputs/image_{image_idx}.txt"
    
    with open(output_file, 'w') as f:
        for q_idx, question in enumerate(questions, 1):
            # Prepare the message for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Process inputs
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            
            # Generate answer
            generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
            answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Write to file in the correct format
            f.write(f"Q{q_idx}: {question}\n")
            f.write(f"A{q_idx}: {answer}\n")

# Process all images from test set
print(f"Starting to process CIFAR-10 test set images...")
for idx, (image, _) in enumerate(tqdm(testset)):
    process_image(image, idx)
    #if idx >= 1000:  # Process first 1000 test images
    #    break

print("Processing complete! Check the SigLIP_Training/qa_outputs directory for results.") 