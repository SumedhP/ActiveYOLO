import os 
import zipfile
from tqdm import tqdm

if __name__ == "__main__":
    input_directory = 'raw_images'
    output_zip_file = input_directory + '.zip'
    
    files = []
    for root, _, filenames in os.walk(input_directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    
    with zipfile.ZipFile(output_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in tqdm(files, desc='Zipping files', unit='file'):
            zipf.write(file, os.path.relpath(file, input_directory))
