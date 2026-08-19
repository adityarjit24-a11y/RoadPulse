import os
from collections import Counter

def validate_yolo_dataset(base_path):
    splits = ['train', 'valid', 'test']
    total_class_counts = Counter()
    
    print(f"Scanning dataset at: {base_path}\n")
    
    for split in splits:
        print(f"--- Checking '{split}' folder ---")
        image_dir = os.path.join(base_path, split, 'images')
        label_dir = os.path.join(base_path, split, 'labels')
        
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"  [!] Skipped: Missing 'images' or 'labels' folder in {split}.\n")
            continue
            
        images = set([os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        labels = set([os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')])
        
        print(f"  Images found: {len(images)}")
        print(f"  Labels found: {len(labels)}")
        
        # 1. Missing files check
        images_without_labels = images - labels
        labels_without_images = labels - images
        
        if images_without_labels:
            print(f"  [!] Warning: {len(images_without_labels)} images have no label file.")
        if labels_without_images:
            print(f"  [!] Warning: {len(labels_without_images)} labels have no image file.")
            
        # 2. Bounding Box & Class Check
        class_counts = Counter()
        malformed_lines = 0
        
        for label_name in labels:
            # Only check labels that have a corresponding image
            if label_name not in images:
                continue
                
            label_path = os.path.join(label_dir, label_name + '.txt')
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        malformed_lines += 1
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total_class_counts[class_id] += 1
                        
                        # YOLO format requires normalized coordinates (0.0 to 1.0)
                        x, y, w, h = map(float, parts[1:])
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            malformed_lines += 1
                    except ValueError:
                        malformed_lines += 1

        print(f"  Malformed annotations: {malformed_lines}")
        print("  Class distribution in this split:")
        for class_id, count in sorted(class_counts.items()):
            print(f"    Class {class_id}: {count} boxes")
        print("\n")

    print("=== FINAL DATASET SUMMARY ===")
    print("Total Unique Classes Found:", len(total_class_counts))
    for class_id, count in sorted(total_class_counts.items()):
        print(f"  Class {class_id}: {count} total boxes")

if __name__ == "__main__":
    # Pointing exactly to your dataset location
    validate_yolo_dataset(r"D:\RoadPulse\datasets")