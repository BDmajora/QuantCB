import os
import subprocess
import sys

def run_engine_auto():
    src_dir = "src"
    
    if not os.path.exists(src_dir):
        print(f"Error: {src_dir} directory not found.")
        return

    # Automatically find all operational scripts in src/
    scripts = sorted([f for f in os.listdir(src_dir) if f.endswith(".py") and not f.startswith("__")])

    print("--- QuantCB Source Runner ---")
    for i, file in enumerate(scripts):
        print(f"[{i}] {file}")

    try:
        user_input = input("\nSelect a script to run (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            return
        
        choice = int(user_input)
        
        if 0 <= choice < len(scripts):
            selected_file = scripts[choice]
            
            env = os.environ.copy()
            root_dir = os.getcwd()
            env["PYTHONPATH"] = root_dir + os.pathsep + os.path.join(root_dir, src_dir) + os.pathsep + env.get("PYTHONPATH", "")

            print(f"\n>> Executing src/{selected_file}...")
            subprocess.run([sys.executable, os.path.join(src_dir, selected_file)], env=env)
        else:
            print("Invalid selection.")

    except ValueError:
        print("Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    run_engine_auto()