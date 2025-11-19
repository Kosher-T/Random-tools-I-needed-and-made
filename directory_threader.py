import os

def list_directory(path, indent=0):
    total_count = 0  # Initialize a counter for this level

    try:
        folder_contents = sorted(os.listdir(path))
    except PermissionError:
        print(" " * indent + "[Permission Denied]")
        return 0  # Return 0 as no files were counted

    for item in folder_contents:
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            print(" " * indent + f"[📁] {item}")
            # Recursively call and add the returned count to the total
            total_count += list_directory(full_path, indent + 4)
        else:
            print(" " * indent + f"- {item}")
            # It's a file, so increment the total count
            total_count += 1
    
    return total_count

if __name__ == "__main__":
    directory = input("Enter the directory path: ").strip()

    if os.path.exists(directory) and os.path.isdir(directory):
        print(f"\nDirectory tree for: {directory}\n")
        final_count = list_directory(directory)
        print(f"\nTotal number of files printed: {final_count}")
    else:
        print("Invalid directory path.")