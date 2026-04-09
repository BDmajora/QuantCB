import os
import subprocess
import sys
import shutil

def cleanup_project():
    """Clears out the modelOutput folder to prepare for a fresh run."""
    project_root = os.getcwd()
    output_dir = os.path.join(project_root, "modelOutput")
    
    if not os.path.exists(output_dir):
        print(f"--- modelOutput folder not found at {output_dir} ---")
        return

    print(f"\n WARNING: This will delete everything in {output_dir}")
    confirm = input("Are you sure? (y/n): ").strip().lower()
    
    if confirm == 'y':
        for f in os.listdir(output_dir):
            if f == ".gitignore": continue
            path = os.path.join(output_dir, f)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                print(f"Deleted: {f}")
            except Exception as e:
                print(f"Failed to delete {f}. Reason: {e}")
        print("--- Cleanup Complete ---")
    else:
        print("Cleanup cancelled.")

def run_engine_auto():
    # Use absolute path for the project root
    root_dir = os.path.abspath(os.getcwd())
    src_dir = os.path.join(root_dir, "src")
    models_dir = os.path.join(root_dir, "models")
    
    if not os.path.exists(src_dir):
        print(f"Error: {src_dir} directory not found.")
        return

    scripts = sorted([f for f in os.listdir(src_dir) if f.endswith(".py") and not f.startswith("__")])
    cleanup_index = len(scripts)

    print("\n--- QuantCB Source Runner ---")
    for i, file in enumerate(scripts):
        print(f"[{i}] {file}")
    print(f"[{cleanup_index}] PROJECT CLEANUP (Wipe modelOutput)")

    try:
        user_input = input("\nSelect a script to run (or 'q' to quit): ").strip()
        if user_input.lower() == 'q': return
        
        choice = int(user_input)
        
        if 0 <= choice < len(scripts):
            selected_file = scripts[choice]
            script_path = os.path.join(src_dir, selected_file)
            
            # --- CRITICAL ENVIRONMENT FIX ---
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1" 
            
            # Added models_dir to the PYTHONPATH to resolve the "block" import
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join([root_dir, src_dir, models_dir, current_pythonpath]).strip(os.pathsep)

            print(f"\n>> Executing: {selected_file}...")
            
            # Run from the root directory to keep relative file paths (like modelOutput/) consistent
            subprocess.run([sys.executable, script_path], env=env, cwd=root_dir)
        
        elif choice == cleanup_index:
            cleanup_project()
        else:
            print("Invalid selection.")

    except ValueError:
        print("Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    run_engine_auto()