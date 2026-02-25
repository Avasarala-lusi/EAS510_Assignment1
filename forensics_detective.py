"""
EAS 510 - Digital Forensics Detective
Prototype v1 — Expert System with 3 Rules (Total 100 pts)
"""
import os

from rules import (
    get_basic_image_info,
    rule1_metadata,
    rule2_color_distribution,
    rule3_visual_similarity,
)

# Try to import Rule 4 — only available in V2
try:
    from rules_v2 import rule4_edge_detection
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False


class SimpleDetective:
    """An expert system that matches modified images to originals."""

    def __init__(self, use_v2=False):
        self.targets = {}
        self.use_v2 = use_v2 and V2_AVAILABLE  # only use V2 if available

    def register_targets(self, folder):
        """Load original images and compute signatures."""
        print(f"Loading targets from: {folder}")

        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(folder, filename)
                file_size = os.stat(filepath).st_size
                basic_info = get_basic_image_info(filepath)

                self.targets[filename] = {
                    "path": filepath,
                    "size": file_size,
                    **basic_info,
                }
                print(f"  Registered: {filename} ({file_size} bytes)")

        print(f"Total targets: {len(self.targets)}\n")

    def find_best_match(self, input_image_path):
        """
        Compare input image against all registered targets using rules.
        V1: 3 rules, max 100 pts, threshold 60
        V2: 4 rules, max 120 pts, threshold 72
        """
        basename = os.path.basename(input_image_path)
        input_info = get_basic_image_info(input_image_path)
        results = []

        # ---- Loop: run rules for every target, collect results ----
        for target_name, target_info in self.targets.items():
            r1_score, r1_fired, r1_ev = rule1_metadata(
                target_info, input_image_path, input_info=input_info
            )
            r2_score, r2_fired, r2_ev = rule2_color_distribution(
                target_info, input_image_path
            )
            r3_score, r3_fired, r3_ev = rule3_visual_similarity(
                target_info, input_image_path
            )

            if self.use_v2:
                r4_score, r4_fired, r4_ev = rule4_edge_detection(
                    target_info, input_image_path
                )
                total = r1_score + r2_score + r3_score + r4_score  # max 120
            else:
                r4_score, r4_fired, r4_ev = 0, False, "Edge score 0.00"
                total = r1_score + r2_score + r3_score              # max 100

            results.append({
                "target":   target_name,
                "total":    total,
                "r1_score": r1_score, "r1_fired": r1_fired, "r1_ev": r1_ev,
                "r2_score": r2_score, "r2_fired": r2_fired, "r2_ev": r2_ev,
                "r3_score": r3_score, "r3_fired": r3_fired, "r3_ev": r3_ev,
                "r4_score": r4_score, "r4_fired": r4_fired, "r4_ev": r4_ev,
            })
        # ---- End of loop ----

        # Pick best candidate across ALL targets
        results.sort(key=lambda x: x["total"], reverse=True)
        best = results[0]
        # print(f"DEBUG raw: r1={best['r1_score']} r2={best['r2_score']} r3={best['r3_score']} r4={best['r4_score']} total={best['total']}")
        # Visual confirmation bonus — Rule 3 strongly confirms visual match
        if self.use_v2:
            max_score = 120
            if best["r3_score"] >= 3:
                best["total"] = min(120, best["total"] + 12)
            if best["r1_score"] >= 10 and best["r2_score"] >= 8:
                best["total"] = min(max_score, best["total"] + 10)
            is_match = best["total"] >= 62
            
        else:
            max_score = 100
            if best["r3_score"] >= 10:
                best["total"] = min(100, best["total"] + 12)
            is_match = best["total"] >= 60
            

        # Zero out scores for rejected images
        if not is_match:
            best["r1_score"] = 0
            best["r2_score"] = 0
            best["r3_score"] = 0
            best["r4_score"] = 0
            best["total"]    = 0
            best["r1_fired"] = False
            best["r2_fired"] = False
            best["r3_fired"] = False
            best["r4_fired"] = False
            best["r1_ev"]    = "Size ratio 0.00"
            best["r2_ev"]    = "Correlation 0.00"
            best["r3_ev"]    = "Match score 0.00"
            best["r4_ev"]    = "Edge score 0.00"

        # Build output lines
        lines = [f"Processing: {basename}"]
        lines.append(
            f"  Rule 1 (Metadata): "
            f"{'FIRED' if best['r1_fired'] else 'NO MATCH'} - "
            f"{best['r1_ev']} -> {best['r1_score']}/30 points"
        )
        lines.append(
            f"  Rule 2 (Histogram): "
            f"{'FIRED' if best['r2_fired'] else 'NO MATCH'} - "
            f"{best['r2_ev']} -> {best['r2_score']}/30 points"
        )
        lines.append(
            f"  Rule 3 (Template): "
            f"{'FIRED' if best['r3_fired'] else 'NO MATCH'} - "
            f"{best['r3_ev']} -> {best['r3_score']}/40 points"
        )

        if self.use_v2:
            lines.append(
                f"  Rule 4 (Edge Detection): "
                f"{'FIRED' if best['r4_fired'] else 'NO MATCH'} - "
                f"{best['r4_ev']} -> {best['r4_score']}/20 points"
            )

        if is_match:
            lines.append(
                f"Final Score: {best['total']}/{max_score} -> MATCH to {best['target']}"
            )
        else:
            lines.append(
                f"Final Score: {best['total']}/{max_score} -> REJECTED"
            )

        output = "\n".join(lines)
        print(output)
        return output, is_match, best["target"] if is_match else None




if __name__ == "__main__":
    print("=" * 50)
    print("SimpleDetective - Prototype v0.1")
    print("=" * 50)

    detective = SimpleDetective(use_v2=True) # whether use V2
    detective.register_targets("originals")

    print("\n" + "=" * 50)
    print("TESTING")
    print("=" * 50)

    test_images = [
        "random/random_08.jpg",
        "modified_images/modified_00_crop_25pct.jpg",
    ]

    folder = "hard"
    images = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    for img in images:
        if os.path.exists(img):
            detective.find_best_match(img)

    print("\n" + "=" * 50)
    print("PROTOTYPE COMPLETE!")
    print("=" * 50)