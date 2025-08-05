#!/bin/bash

# Usage: ./folder_tree.sh /path/to/root

show_tree() {
    local dir="$1"
    local prefix="$2"
    local folders=()
    local files=()
    
    # Separate folders and files
    while IFS= read -r -d '' entry; do
        if [ -d "$entry" ]; then
            folders+=("$entry")
        else
            files+=("$entry")
        fi
    done < <(find "$dir" -mindepth 1 -maxdepth 1 -print0 | sort -z)
    
    # Always show ALL folders
    for folder in "${folders[@]}"; do
        name=$(basename "$folder")
        echo "${prefix}├── $name"
        show_tree "$folder" "$prefix│   "
    done
    
    # Handle files with truncation if more than 3
    local total_files=${#files[@]}
    local max_files=3
    
    if (( total_files <= max_files )); then
        # Show all files if 3 or fewer
        for file in "${files[@]}"; do
            name=$(basename "$file")
            echo "${prefix}├── $name"
        done
    else
        # Show first 2 files
        for i in {0..1}; do
            name=$(basename "${files[i]}")
            echo "${prefix}├── $name"
        done
        
        # Show truncation indicator
        echo "${prefix}│   ⋮ ($((total_files-2)) more files)"
        
        # Show last file
        name=$(basename "${files[$((total_files-1))]}")
        echo "${prefix}├── $name"
    fi
}

root="${1:-.}"
echo "$root"
show_tree "$root" ""