import json
import os

path_a_features = r"c:\Users\Bureau\Challenge_data\Challenge-data\challenge_code\results\experiments\251209-18\features.json"
path_b_features = r"c:\Users\Bureau\Challenge_data\Challenge-data\challenge_code\results\experiments\archive\e3e1d1f709\features.json"

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def compare_lists(list_a, list_b, name_a="New (251209-18)", name_b="Archive (e3e1d1f709)"):
    set_a = set(list_a)
    set_b = set(list_b)
    
    added = set_a - set_b
    removed = set_b - set_a
    
    print(f"--- Comparison of Features ({name_a} vs {name_b}) ---")
    print(f"Total in {name_a}: {len(list_a)}")
    print(f"Total in {name_b}: {len(list_b)}")
    
    if added:
        print(f"\nAdded in {name_a} ({len(added)}):")
        for item in sorted(added):
            print(f"  + {item}")
            
    if removed:
        print(f"\nRemoved in {name_a} (Present in {name_b}) ({len(removed)}):")
        for item in sorted(removed):
            print(f"  - {item}")
            
    if not added and not removed:
        print("\nNo differences in feature sets.")

try:
    f_a = load_json(path_a_features)
    f_b = load_json(path_b_features)
    compare_lists(f_a, f_b)
except Exception as e:
    print(f"Error comparing features: {e}")
