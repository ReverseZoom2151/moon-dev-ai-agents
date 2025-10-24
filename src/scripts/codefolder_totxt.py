# Standard library imports
import glob
import os

# Standard library from imports
from datetime import datetime
from pathlib import Path

# Calculate project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Constants - Now using relative paths from project root
INPUT_FOLDER = str(PROJECT_ROOT / "src")  # Default to scanning src folder
OUTPUT_FOLDER = str(PROJECT_ROOT / "src" / "data" / "codetotext")

# File extensions to include
CODE_EXTENSIONS = ['.py', '.ipynb', '.js', '.html', '.css', '.md', '.txt']

def process_folder_to_txt():
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get the last part of the input folder path for the filename
    folder_name = os.path.basename(INPUT_FOLDER)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_FOLDER, f"{folder_name}_code_dump_{timestamp}.txt")
    
    print(f"🌙 Moon Dev's Code Collector starting up! 🌙")
    print(f"📂 Scanning folder: {INPUT_FOLDER}")
    
    # Initialize counter for stats
    total_files = 0
    total_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(f"// Moon Dev's Code Collection - Generated on {datetime.now()}\n")
        outfile.write(f"// May the code be with you! 🌙\n\n")
        
        # Walk through all directories
        for root, dirs, files in os.walk(INPUT_FOLDER):
            for file in files:
                if any(file.endswith(ext) for ext in CODE_EXTENSIONS):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, INPUT_FOLDER)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            
                            # Write file header
                            outfile.write(f"{'='*80}\n")
                            outfile.write(f"File: {relative_path}\n")
                            outfile.write(f"{'='*80}\n\n")
                            outfile.write(content)
                            outfile.write("\n\n")
                            
                            total_files += 1
                            total_lines += len(content.splitlines())
                            
                            print(f"✨ Processed: {relative_path}")
                    except Exception as e:
                        print(f"❌ Error processing {relative_path}: {str(e)}")

    print(f"\n🎉 Moon Dev's Code Collection Complete! 🎉")
    print(f"📊 Stats:")
    print(f"   - Total files processed: {total_files}")
    print(f"   - Total lines of code: {total_lines}")
    print(f"📝 Output saved to: {output_file}")
    print(f"\n💫 Thanks for using Moon Dev's Code Collector! Keep coding! 💫")

if __name__ == "__main__":
    process_folder_to_txt()
