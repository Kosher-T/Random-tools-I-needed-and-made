import os

def list_directory(path, indent=0):
    try:
        # List all items in the given path
        items = sorted(os.listdir(path))
    except PermissionError:
        print(" " * indent + "[Permission Denied]")
        return
    
    for item in items:
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            print(" " * indent + f"[📁] {item}")
            list_directory(full_path, indent + 4)  # Recurse into subfolder
        else:
            print(" " * indent + f"- {item}")

if __name__ == "__main__":
    directory = input("Enter the directory path: ").strip()
    
    if os.path.exists(directory) and os.path.isdir(directory):
        print(f"\nDirectory tree for: {directory}\n")
        list_directory(directory)
    else:
        print("Invalid directory path.")