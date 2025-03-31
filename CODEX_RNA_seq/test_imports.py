"""
Script to test imports in all Python files with a timeout.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def test_file_imports(file_path, timeout=5):
    """Test imports in a Python file with timeout."""
    try:
        # Run the file with Python and capture output
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {file_path}: Imports successful")
            return True
        else:
            print(f"❌ {file_path}: Import error")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⚠️ {file_path}: Timeout after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Error: {str(e)}")
        return False

def add_import_setup(file_path):
    """Add import setup code to a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if setup code already exists
    if "sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))" in content:
        print(f"ℹ️ {file_path}: Setup code already exists")
        return
    
    # Add setup code after the first import statement
    setup_code = """
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
    
    # Find the first import statement
    lines = content.split('\n')
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            insert_pos = i
            break
    
    # Insert setup code
    lines.insert(insert_pos, setup_code)
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✨ {file_path}: Added import setup code")

def main():
    """Main function to process all Python files."""
    current_dir = Path(__file__).parent
    python_files = list(current_dir.glob('*.py'))
    
    print(f"Found {len(python_files)} Python files to process")
    
    for file_path in python_files:
        if file_path.name == 'test_imports.py':
            continue
            
        print(f"\nProcessing {file_path.name}...")
        add_import_setup(file_path)
        test_file_imports(file_path)

if __name__ == "__main__":
    main() 