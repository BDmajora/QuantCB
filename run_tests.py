import os
import subprocess
import sys

def run_tests():
    test_dir = "tests"
    src_dir = "src"
    
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} directory not found.")
        return

    # Gather all python files starting with 'test_'
    test_files = sorted([f for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py")])

    if not test_files:
        print("No test files found in /tests/")
        return

    print("--- QuantCB Test Suite ---")
    for i, file in enumerate(test_files):
        print(f"[{i}] {file}")
    print(f"[{len(test_files)}] RUN ALL TESTS")

    try:
        user_input = input("\nSelect a test number (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            return
        
        choice = int(user_input)
        
        # Inject root and src into PYTHONPATH
        # This allows tests to import from 'models' and 'src' seamlessly
        env = os.environ.copy()
        root_dir = os.getcwd()
        full_src_path = os.path.join(root_dir, src_dir)
        
        paths = [root_dir, full_src_path]
        env["PYTHONPATH"] = os.pathsep.join(paths) + os.pathsep + env.get("PYTHONPATH", "")

        def execute_test(filename):
            print(f"\n>> Executing {filename}...")
            # Use sys.executable to maintain the current venv
            result = subprocess.run([sys.executable, os.path.join(test_dir, filename)], env=env)
            if result.returncode != 0:
                print(f"{filename} FAILED with exit code {result.returncode}")
            else:
                print(f"{filename} PASSED")

        if choice == len(test_files):
            print("\nRunning full suite...")
            for file in test_files:
                execute_test(file)
        elif 0 <= choice < len(test_files):
            execute_test(test_files[choice])
        else:
            print("Invalid selection.")

    except ValueError:
        print("Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    run_tests()