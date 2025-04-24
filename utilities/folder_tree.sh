#!/bin/bash

# Usage: ./folder_tree.sh /path/to/root

show_tree() {
    local dir="$1"
    local prefix="$2"
    local entries=()
    
    # Read only non-hidden entries
    while IFS= read -r -d '' entry; do
        entries+=("$entry")
    done < <(find "$dir" -mindepth 1 -maxdepth 1 -print0 | sort -z)

    local total=${#entries[@]}
    local max_display=5

    if (( total <= max_display )); then
        for entry in "${entries[@]}"; do
            name=$(basename "$entry")
            echo "${prefix}├── $name"
            [ -d "$entry" ] && show_tree "$entry" "$prefix│   "
        done
    else
        for i in {0..1}; do
            name=$(basename "${entries[i]}")
            echo "${prefix}├── $name"
            [ -d "${entries[i]}" ] && show_tree "${entries[i]}" "$prefix│   "
        done

        echo "${prefix}│   ⋮"

        for i in $(seq $((total-2)) $((total-1))); do
            name=$(basename "${entries[i]}")
            echo "${prefix}├── $name"
            [ -d "${entries[i]}" ] && show_tree "${entries[i]}" "$prefix│   "
        done
    fi
}

root="${1:-.}"
echo "$root"
show_tree "$root" ""
