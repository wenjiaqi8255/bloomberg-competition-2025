#!/usr/bin/env python3
"""
Monitor FF5 Training with FDR Correction

This script monitors the training log and extracts FDR correction results
when training completes.
"""

import re
import time
import sys
from pathlib import Path

LOG_FILE = "/tmp/ff5_training_fdr.log"

def monitor_log(log_file: str, patterns: list):
    """
    Monitor log file for specific patterns and print when found.

    Args:
        log_file: Path to log file
        patterns: List of regex patterns to search for
    """
    print(f"🔍 Monitoring {log_file} for FDR results...")
    print(f"   Looking for: {patterns}")
    print("   Press Ctrl+C to stop monitoring\n")

    last_position = 0
    found_patterns = {pattern: False for pattern in patterns}

    try:
        while True:
            with open(log_file, 'r') as f:
                # Go to last position
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()

                # Check each line for patterns
                for line in new_lines:
                    for pattern in patterns:
                        if not found_patterns[pattern] and re.search(pattern, line, re.IGNORECASE):
                            print(f"\n✅ FOUND: {line.strip()}")
                            found_patterns[pattern] = True

                            # If all patterns found, show full context
                            if all(found_patterns.values()):
                                print("\n" + "="*70)
                                print("🎉 ALL FDR OUTPUT FOUND! Showing full context...")
                                print("="*70 + "\n")
                                show_fdr_context(log_file)
                                return

            # If all patterns found, we're done
            if all(found_patterns.values()):
                break

            # Wait before checking again
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped by user")
        sys.exit(0)


def show_fdr_context(log_file: str, context_lines: int = 50):
    """
    Show FDR correction context from log file.

    Args:
        log_file: Path to log file
        context_lines: Number of lines to show before and after
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Find FDR section
    fdr_start = None
    for i, line in enumerate(lines):
        if re.search(r'Benjamini-Hochberg.*FDR.*Correction', line, re.IGNORECASE):
            fdr_start = i
            break

    if fdr_start is None:
        print("⚠️  FDR section not found in log")
        print("Training may still be in progress. Full log tail:")
        print("".join(lines[-context_lines:]))
        return

    # Find end of FDR section (next major section or end of file)
    fdr_end = fdr_start + 200  # Max 200 lines
    for i in range(fdr_start, min(fdr_start + 200, len(lines))):
        if re.search(r'^={70,}', lines[i]):
            fdr_end = i + 1
            break

    # Show FDR section
    print("".join(lines[max(0, fdr_start-5):min(fdr_end, len(lines))]))


def check_training_status(log_file: str):
    """Check current training status."""
    with open(log_file, 'r') as f:
        content = f.read()

    if re.search(r'Benjamini-Hochberg.*FDR', content, re.IGNORECASE):
        return "✅ FDR Results Available"
    elif re.search(r'Training pipeline|Step 1: Loading', content, re.IGNORECASE):
        return "🔄 Training in Progress"
    elif re.search(r'Fetching historical data|Successfully fetched', content, re.IGNORECASE):
        return "📥 Fetching Data"
    else:
        return "❓ Unknown Status"


if __name__ == "__main__":
    # Check if log file exists
    if not Path(LOG_FILE).exists():
        print(f"❌ Log file not found: {LOG_FILE}")
        print(f"   Training may not have started yet.")
        sys.exit(1)

    # Show current status
    status = check_training_status(LOG_FILE)
    print(f"Current Status: {status}\n")

    # Patterns to look for
    fdr_patterns = [
        r'Benjamini-Hochberg.*FDR.*Correction',
        r'FDR Level.*Q',
        r'Significant Features.*after FDR',
        r'p_value_fdr|adj_p',
    ]

    # Monitor for FDR results
    monitor_log(LOG_FILE, fdr_patterns)

    print("\n" + "="*70)
    print("📊 Monitoring Complete!")
    print("="*70)
    print(f"\nFull log available at: {LOG_FILE}")
    print("To view full log: tail -n 100 /tmp/ff5_training_fdr.log")
