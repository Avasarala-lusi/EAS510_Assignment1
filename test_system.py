import os
import sys
phase_1    = False   # → results_v1.txt     (modified + random, V1)
phase_hard = False   # → results_v1_hard.txt (hard only, V1)
phase_2    = True   # → results_v2.txt      (modified + hard + random, V2)

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE      = os.path.join(SCRIPT_DIR, "results_v1.txt")
HARD_RESULTS_FILE = os.path.join(SCRIPT_DIR, "results_v1_hard.txt")
RESULTS_V2_FILE   = os.path.join(SCRIPT_DIR, "results_v2.txt")

sys.path.insert(0, SCRIPT_DIR)
from forensics_detective import SimpleDetective


def run_folder(detective, folder):
    """Run detective on all images in a folder, return output lines."""
    outputs = []
    image_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    for filename in image_files:
        path = os.path.join(folder, filename)
        output, _, _ = detective.find_best_match(path)
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    # ---- Flags: set one to True at a time ----
    phase_1 = False   # runs modified_images/ + random/ → results_v1.txt
    phase_hard = False # runs hard/ only              → results_v1_hard.txt
    phase_2 = True    # runs modified_images/ + hard/ + random/ → results_v2.txt

    # Phase 2 uses V2 (4 rules), others use V1 (3 rules)
    detective = SimpleDetective(use_v2=phase_2)

    # Suppress register_targets printing
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    detective.register_targets(os.path.join(SCRIPT_DIR, "originals"))
    sys.stdout = sys.__stdout__
    devnull.close()

    all_outputs = []

    if phase_1:
        all_outputs.extend(run_folder(detective, os.path.join(SCRIPT_DIR, "modified_images")))
        all_outputs.extend(run_folder(detective, os.path.join(SCRIPT_DIR, "random")))
        with open(RESULTS_FILE, "w") as f:
            f.write("\n".join(all_outputs) + "\n")
        print(f"Saved to: {RESULTS_FILE}")

    elif phase_hard:
        all_outputs.extend(run_folder(detective, os.path.join(SCRIPT_DIR, "hard")))
        with open(HARD_RESULTS_FILE, "w") as f:
            f.write("\n".join(all_outputs) + "\n")
        print(f"Saved to: {HARD_RESULTS_FILE}")

    elif phase_2:
        all_outputs.extend(run_folder(detective, os.path.join(SCRIPT_DIR, "modified_images")))
        all_outputs.extend(run_folder(detective, os.path.join(SCRIPT_DIR, "hard")))
        all_outputs.extend(run_folder(detective, os.path.join(SCRIPT_DIR, "random")))
        with open(RESULTS_V2_FILE, "w") as f:
            f.write("\n".join(all_outputs) + "\n")
        print(f"Saved to: {RESULTS_V2_FILE}")