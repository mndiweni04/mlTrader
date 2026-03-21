# verify_integrity.py
import os
import hashlib
import json
import argparse
import sys

MANIFEST_FILE = "integrity_manifest.json"

# The definitive list of core system files that must be audited
CORE_FILES = [
    "dynamic_features.py",
    "process_data.py",
    "train_model.py",
    "evaluate_model.py",
    "predict.py",
    "run_trader.py",
    "walk_forward_backtest.py",
    "monitor_trades.py",
    "explain_model.py",
    "visualize_results.py",
    "retrain_pipeline.sh"
]

def calculate_normalized_hash(filepath):
    """
    Calculates a SHA-256 hash after normalizing line endings.
    This prevents false positives between Windows (CRLF) and Linux (LF).
    """
    if not os.path.exists(filepath):
        return None

    hasher = hashlib.sha256()
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        # Normalize all line endings to standard Unix \n
        normalized_content = content.replace('\r\n', '\n').strip()
        hasher.update(normalized_content.encode('utf-8'))
    
    return hasher.hexdigest()

def generate_manifest():
    """Scans core files and generates the baseline integrity manifest."""
    manifest = {}
    missing_files = []

    print("="*50)
    print(" 🛡️ GENERATING INTEGRITY MANIFEST")
    print("="*50)

    for filename in CORE_FILES:
        file_hash = calculate_normalized_hash(filename)
        if file_hash:
            manifest[filename] = file_hash
            print(f" [LOCKED] {filename} -> {file_hash[:8]}...")
        else:
            missing_files.append(filename)
            print(f" [MISSING] {filename}")

    with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)
    
    print("\n✅ Manifest saved to", MANIFEST_FILE)
    if missing_files:
        print(f"⚠️  WARNING: {len(missing_files)} files were missing and not hashed.")

def verify_system():
    """Compares current files against the locked integrity manifest."""
    print("="*50)
    print(" 🔍 VERIFYING SYSTEM INTEGRITY")
    print("="*50)

    if not os.path.exists(MANIFEST_FILE):
        print(f"❌ Error: {MANIFEST_FILE} not found. Run with --generate first.")
        sys.exit(1)

    with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    passed = 0
    failed = 0

    for filename, expected_hash in manifest.items():
        current_hash = calculate_normalized_hash(filename)
        
        if not current_hash:
            print(f" ❌ [FAIL] {filename}: FILE MISSING")
            failed += 1
        elif current_hash != expected_hash:
            print(f" ❌ [FAIL] {filename}: HASH MISMATCH (File altered)")
            failed += 1
        else:
            print(f" ✅ [PASS] {filename}")
            passed += 1

    print("-" * 50)
    if failed == 0:
        print(f"🟢 SYSTEM SECURE: All {passed} files passed integrity check.")
        sys.exit(0)
    else:
        print(f"🔴 INTEGRITY COMPROMISED: {failed} files failed verification.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Trader SHA-256 Integrity Monitor")
    parser.add_argument("--generate", action="store_true", help="Lock current file states and generate manifest")
    parser.add_argument("--verify", action="store_true", help="Verify current files against the manifest")
    
    args = parser.parse_args()

    if args.generate:
        generate_manifest()
    elif args.verify:
        verify_system()
    else:
        # Default behavior if no arguments are passed
        if os.path.exists(MANIFEST_FILE):
            verify_system()
        else:
            parser.print_help()