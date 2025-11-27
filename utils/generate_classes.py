import os

def generate_classes_file():
    data_processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data_processed')
    output_file = os.path.join(os.path.dirname(__file__), '..', 'classes.txt')
    
    if not os.path.exists(data_processed_dir):
        print(f"Error: {data_processed_dir} does not exist.")
        return

    folders = sorted([d for d in os.listdir(data_processed_dir) if os.path.isdir(os.path.join(data_processed_dir, d))])
    
    with open(output_file, 'w') as f:
        for folder in folders:
            f.write(f"{folder}\n")
            
    print(f"Successfully generated {output_file} with {len(folders)} classes.")

if __name__ == "__main__":
    generate_classes_file()
