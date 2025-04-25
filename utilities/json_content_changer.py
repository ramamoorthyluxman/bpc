#!/usr/bin/env python3
import os
import json
import argparse
import re
from typing import Tuple, List

def search_and_replace_in_json_files(folder_path: str, search_word: str, replacement_word: str) -> Tuple[int, int]:
    """
    Search and replace text in all JSON files in a directory
    
    Args:
        folder_path: Path to the folder containing JSON files
        search_word: Word to search for
        replacement_word: Word to replace with
        
    Returns:
        Tuple containing (total replacements made, number of files changed)
    """
    print(f"Searching for '{search_word}' and replacing with '{replacement_word}' in folder: {folder_path}")
    
    total_replacements = 0
    files_changed = 0
    
    try:
        # Get all files in the directory
        files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count occurrences
                occurrences = len(re.findall(re.escape(search_word), content))
                
                if occurrences > 0:
                    # Perform replacement with escaped search term to handle special regex characters
                    new_content = re.sub(re.escape(search_word), replacement_word, content)
                    
                    # Write the updated content back to the file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    total_replacements += occurrences
                    files_changed += 1
                    
                    print(f"Modified {file}: {occurrences} replacement(s) made")
            
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print(f"\nSummary: Made {total_replacements} replacement(s) across {files_changed} file(s)")
    return total_replacements, files_changed


def recursive_search_and_replace(folder_path: str, search_word: str, replacement_word: str, recursive: bool = False) -> Tuple[int, int]:
    """
    Search and replace text in all JSON files in a directory and optionally in subdirectories
    
    Args:
        folder_path: Path to the folder containing JSON files
        search_word: Word to search for
        replacement_word: Word to replace with
        recursive: Whether to search in subdirectories
        
    Returns:
        Tuple containing (total replacements made, number of files changed)
    """
    total_replacements = 0
    files_changed = 0
    
    # Process current directory
    replacements, changed = search_and_replace_in_json_files(folder_path, search_word, replacement_word)
    total_replacements += replacements
    files_changed += changed
    
    # Process subdirectories if recursive flag is set
    if recursive:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                sub_replacements, sub_changed = recursive_search_and_replace(
                    item_path, search_word, replacement_word, recursive=True
                )
                total_replacements += sub_replacements
                files_changed += sub_changed
    
    return total_replacements, files_changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search and replace text in JSON files')
    parser.add_argument('folder_path', help='Path to folder containing JSON files')
    parser.add_argument('search_word', help='Word to search for')
    parser.add_argument('replacement_word', help='Word to replace with')
    parser.add_argument('-r', '--recursive', action='store_true', help='Search in subdirectories')
    
    args = parser.parse_args()
    
    if args.recursive:
        recursive_search_and_replace(args.folder_path, args.search_word, args.replacement_word, recursive=True)
    else:
        search_and_replace_in_json_files(args.folder_path, args.search_word, args.replacement_word)