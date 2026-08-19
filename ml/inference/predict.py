import os
import json
from ultralytics import YOLO

def run_inference(image_path, model_path):
    print(f"Loading model from {model_path}...")
    # Load your local trained model
    model = YOLO(model_path)
    
    print(f"Running inference on {image_path}...")
    # conf=0.15 rakha hai kyunki hamara MVP 10% data par bana hai
    results = model(image_path, conf=0.1)
    
    detections = []
    output_path = ""
    
    for r in results:
        # 1. Annotated image save karna (boxes ke sath)
        save_dir = r"D:\RoadPulse\ml\inference\output"
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, "detected_image.jpg")
        r.save(filename=output_path)
        
        # 2. JSON data extract karna (FastAPI backend ke liye)
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist() 
            
            detections.append({
                "class": cls_name,
                "confidence": round(conf, 2),
                "bbox": [round(x, 1) for x in xyxy]
            })

    output_json = {
        "image": os.path.basename(image_path),
        "detections": detections
    }

    print("\n=== AI DETECTIONS JSON ===")
    print(json.dumps(output_json, indent=2))
    print(f"\n✅ Annotated image saved to: {output_path}")

if __name__ == "__main__":
    test_img = r"D:\RoadPulse\test_image.jpg"
    model_weight = r"D:\RoadPulse\ml\best.pt"
    
    if not os.path.exists(test_img):
        print(f"❌ Error: Image nahi mili. Kripya image ko yahan rakhein: {test_img}")
    elif not os.path.exists(model_weight):
        print(f"❌ Error: Model nahi mila. Kripya best.pt ko yahan rakhein: {model_weight}")
    else:
        run_inference(test_img, model_weight)