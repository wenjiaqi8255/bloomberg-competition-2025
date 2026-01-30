#!/usr/bin/env python3
"""
Fix hardcoded 'src.' imports throughout the codebase.

This script replaces all 'from src.trading_system' with 'from trading_system'
across all Python files in the src/ directory.
"""

import os
import re
from pathlib import Path

def fix_src_imports(file_path):
    """Fix hardcoded src imports in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Replace 'from src.trading_system' with 'from trading_system'
        content = re.sub(r'from src\.trading_system', 'from trading_system', content)

        # Also handle 'import src.trading_system'
        content = re.sub(r'import src\.trading_system', 'import trading_system', content)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Fix all Python files in src/ directory."""
    src_dir = Path('/Users/wenjiaqi/Downloads/bloomberg-competition/src')

    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist")
        return

    # Find all Python files with 'from src.' imports
    fixed_count = 0
    total_files = 0

    for py_file in src_dir.rglob('*.py'):
        # Check if file contains 'from src.' or 'import src.'
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                if 'from src.' in content or 'import src.' in content:
                    total_files += 1
                    if fix_src_imports(py_file):
                        fixed_count += 1
                        print(f"✓ Fixed: {py_file.relative_to(src_dir.parent)}")
        except Exception as e:
            print(f"✗ Error reading {py_file}: {e}")

    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} out of {total_files} files")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
