import os
import argparse
import sys

def list_directory(path, indent=0, exclude_dirs=None, exclude_exts=None):
    """
    Recursively lists directory contents, excluding specified folders and extensions.
    """
    # Handle mutable default arguments safely
    if exclude_dirs is None:
        exclude_dirs = set()
    if exclude_exts is None:
        exclude_exts = []

    total_count = 0 

    try:
        folder_contents = sorted(os.listdir(path))
    except PermissionError:
        print(" " * indent + "[Permission Denied]")
        return 0

    for item in folder_contents:
        full_path = os.path.join(path, item)
        
        if os.path.isdir(full_path):
            # CHECK 1: If the folder name is in the exclusion list, skip it entirely
            if item in exclude_dirs:
                continue
            
            print(" " * indent + f"[📁] {item}")
            # Pass the exclusion lists down to the recursive call
            total_count += list_directory(
                full_path, 
                indent + 4, 
                exclude_dirs, 
                exclude_exts
            )
        else:
            # CHECK 2: If the file ends with an excluded extension, skip it
            # We convert the list to a tuple because .endswith() accepts tuples
            if exclude_exts and item.endswith(tuple(exclude_exts)):
                continue
                
            print(" " * indent + f"- {item}")
            total_count += 1
    
    return total_count

if __name__ == "__main__":
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="Recursive directory lister with exclusions.")
    
    # Positional argument: The directory to scan
    parser.add_argument("path", help="The directory path to scan")
    
    # Optional argument: Folders to exclude
    parser.add_argument(
        "-d", "--exclude-dirs", 
        nargs="*",  # Accepts 0 or more arguments
        default=[], 
        help="List of folder names to ignore (e.g. node_modules __pycache__)"
    )
    
    # Optional argument: Extensions to exclude
    parser.add_argument(
        "-e", "--exclude-exts", 
        nargs="*", 
        default=[], 
        help="List of file extensions to ignore (e.g. .pyc .txt .log)"
    )

    args = parser.parse_args()
    directory = args.path

    if os.path.exists(directory) and os.path.isdir(directory):
        print(f"\nDirectory tree for: {directory}")
        if args.exclude_dirs:
            print(f"Ignoring folders: {', '.join(args.exclude_dirs)}")
        if args.exclude_exts:
            print(f"Ignoring extensions: {', '.join(args.exclude_exts)}")
        print("-" * 40 + "\n")

        final_count = list_directory(
            directory, 
            exclude_dirs=set(args.exclude_dirs), # Use set for faster lookups
            exclude_exts=args.exclude_exts
        )
        print(f"\nTotal number of files printed: {final_count}")
    else:
        print("Invalid directory path.")