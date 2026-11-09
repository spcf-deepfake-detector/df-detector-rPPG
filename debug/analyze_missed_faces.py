import json
from pathlib import Path

# Directory containing extracted face images
root = Path("extracted_faces")

total_images = 0
total_detected = 0
total_missed = 0

# Find every detection_results.json
for result_file in root.rglob("detection_results.json"):

    with open(result_file, "r", encoding="utf-8") as file:
        results = json.load(file)

    missed = [
        result for result in results
        if not result["detected"]
    ]

    detected = len(results) - len(missed)

    total_images += len(results)
    total_detected += detected
    total_missed += len(missed)

    if missed:

        print(f"\nSequence: {result_file.parent}")
        print(f"Total images: {len(results)}")
        print(f"Detected: {detected}")
        print(f"Missed: {len(missed)}")

        print("\nMissed image paths:")

        for result in missed:
            print(result["image"])

# Overall summary
print("\n========== OVERALL SUMMARY ==========")

print(f"Total images: {total_images}")
print(f"Detected: {total_detected}")
print(f"Missed: {total_missed}")

if total_images > 0:
    print(
        f"Detection rate: "
        f"{total_detected / total_images * 100:.2f}%"
    )

print("=====================================")