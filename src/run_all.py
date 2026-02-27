import subprocess
import sys
def run_script(script_name):
    print(f"\nRunning {script_name}...")
    result = subprocess.run([sys.executable, f"src/{script_name}"])
    
    if result.returncode != 0:
        print(f"Error while running {script_name}")
        sys.exit(1)
    else:
        print(f"{script_name} completed successfully.")

if __name__ == "__main__":
    print("Starting full pipeline...\n")
    run_script("data_cleaning.py")
    run_script("eda.py")
    print("\nAll steps completed successfully!")
