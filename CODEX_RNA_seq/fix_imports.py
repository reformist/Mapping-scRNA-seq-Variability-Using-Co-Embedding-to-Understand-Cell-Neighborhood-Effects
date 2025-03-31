"""
Script to fix imports in all Python files by removing os.chdir and cleaning up imports.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import re
from pathlib import Path

def fix_imports(file_path):
    """Fix imports in a Python file by removing os.chdir and cleaning up imports."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove os.chdir lines
    content = re.sub(r"os\.chdir\('CODEX_RNA_seq'\)\n?", "", content)
    
    # Remove duplicate imports
    content = re.sub(r"import os\n+import os", "import os", content)
    content = re.sub(r"import sys\n+import sys", "import sys", content)
    
    # Remove parent_folder setup if it exists
    content = re.sub(r"parent_folder = os\.path\.abspath\(os\.path\.join\(os\.getcwd\(\), '\.\.'\)\)\n+sys\.path\.append\(parent_folder\)\n?", "", content)
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✨ Fixed imports in {file_path}")

def main():
    """Main function to process all Python files."""
    current_dir = Path(__file__).parent
    python_files = list(current_dir.glob('*.py'))
    
    print(f"Found {len(python_files)} Python files to process")
    
    for file_path in python_files:
        if file_path.name in ['test_imports.py', 'fix_imports.py']:
            continue
            
        print(f"\nProcessing {file_path.name}...")
        fix_imports(file_path)

if __name__ == "__main__":
    main() 