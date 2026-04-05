import os
import subprocess
import sys
import shutil

def cleanup_project():
    """Clears out the modelOutput folder to prepare for a fresh run."""
    project_root = os.getcwd()
    # Ensure this matches the OUTPUT_DIR in your config
    output_dir = os.path.join(project_root, "modelOutput")
    
    if not os.path.exists(output_dir):
        print(f"--- modelOutput folder not found at {output_dir} ---")
        return

    print(f"\n WARNING: This will delete everything in {output_dir}")
    confirm = input("Are you sure? (y/n): ").strip().lower()
    
    if confirm == 'y':
        files = os.listdir(output_dir)
        for f in files:
            if f == ".gitignore":
                continue
            
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
    src_dir = "src"
    
    if not os.path.exists(src_dir):
        print(f"Error: {src_dir} directory not found.")
        return

    # Automatically find all operational scripts in src/
    scripts = sorted([f for f in os.listdir(src_dir) if f.endswith(".py") and not f.startswith("__")])
    
    cleanup_index = len(scripts)

    print("\n--- QuantCB Source Runner ---")
    for i, file in enumerate(scripts):
        print(f"[{i}] {file}")
    print(f"[{cleanup_index}] PROJECT CLEANUP (Wipe modelOutput)")

    try:
        user_input = input("\nSelect a script to run (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            return
        
        choice = int(user_input)
        
        if 0 <= choice < len(scripts):
            selected_file = scripts[choice]
            
            # --- FIX: Setup Environment for Real-Time Output ---
            env = os.environ.copy()
            
            # This flag prevents Python from buffering stdout. 
            # Without this, you won't see logs until the buffer fills up.
            env["PYTHONUNBUFFERED"] = "1" 
            
            root_dir = os.getcwd()
            # Construct PYTHONPATH so local imports work correctly
            env["PYTHONPATH"] = (
                root_dir + os.pathsep + 
                os.path.join(root_dir, src_dir) + os.pathsep + 
                env.get("PYTHONPATH", "")
            )

            print(f"\n>> Executing src/{selected_file}...")
            
            # subprocess.run blocks execution until the child script finishes
            subprocess.run([sys.executable, os.path.join(src_dir, selected_file)], env=env)
        
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