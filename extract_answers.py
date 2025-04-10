import os
import re
import glob

def extract_assistant_answers(input_file):
    """Extract the text after 'Assistant:' from the input file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content by "Assistant:" to get all sections after it
    sections = content.split("Assistant:")
    
    # Process each section to get clean answers
    answers = []
    for section in sections[1:]:  # Skip the first split as it's before first "Assistant:"
        # Get text up to next "Q" or "User:" or end of string
        answer = section.split("Q")[0].split("User:")[0].strip()
        if answer:
            answers.append(answer)
    
    return answers

def process_all_files():
    """Process all image_*.txt files in the qa_outputs directory."""
    # Get all image_*.txt files
    input_files = glob.glob("qa_outputs/image_*.txt")
    
    for input_file in input_files:
        # Extract the base name without extension
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_extr.txt"
        
        # Extract answers
        answers = extract_assistant_answers(input_file)
        
        # Write answers to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, answer in enumerate(answers, 1):
                f.write(f"{answer}\n")
        
        print(f"Processed {input_file} -> {output_file}")

if __name__ == "__main__":
    process_all_files()
    print("Extraction complete! Check the files with '_extr' suffix.") 