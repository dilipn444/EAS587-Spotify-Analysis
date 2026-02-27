import os

def check_path(path):
    if os.path.exists(path):
        print(f"[OK] {path} exists.")
    else:
        print(f"[MISSING] {path} not found.")

if __name__ == "__main__":
    print("Running repository structure check...\n")
    
    check_path("requirements.txt")
    check_path("README.md")
    check_path("data/raw/README.md")
    check_path("src/data_cleaning.py")
    check_path("src/eda.py")
    check_path("reports/figures")
  
    print("\nStructure check complete.")
