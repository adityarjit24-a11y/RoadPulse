import os
import gc
from pydantic import BaseModel
from sqlalchemy import text
from fastapi import HTTPException, FastAPI, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from ultralytics import YOLO
from PIL import Image
import io
from datetime import datetime

# Import StatusHistory to track our timeline
from database import SessionLocal, Report, Detection, Location, StatusHistory 
from priority_engine import compute_priority 
from duplicate_detection import check_for_duplicate, _compute_phash

app = FastAPI(title="RoadPulse API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Image Hosting Setup ---
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
# ---------------------------------

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "Welcome to RoadPulse API! The server is running successfully. 🚀"}

@app.post("/predict")
async def predict_road_damage(
    file: UploadFile = File(...), 
    lat: float = Form(None), 
    lon: float = Form(None), 
    db: Session = Depends(get_db)
):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    img_width, img_height = image.size 
    img_area = img_width * img_height

    if lat is not None and lon is not None:
        def dummy_loader(report_id):
            with open(f"static/uploads/{report_id}.jpg", "rb") as f:
                return f.read()
        
        dup_result = check_for_duplicate(db, image_data, lat, lon, dummy_loader)
        
        if dup_result.is_duplicate:
            existing_report = db.query(Report).filter(Report.id == dup_result.matched_report_id).first()
            if existing_report:
                existing_report.citizen_report_count += 1
                
                priority_breakdown = compute_priority(
                    severity_score=existing_report.priority_score or 50.0,
                    road_type="unknown", 
                    near_sensitive_site=False, 
                    citizen_report_count=existing_report.citizen_report_count,
                    days_unresolved=existing_report.days_unresolved,
                    detection_confidence=0.80 
                )
                existing_report.priority_score = priority_breakdown.total_score
                existing_report.priority_band = priority_breakdown.band
                
                db.commit()
                db.refresh(existing_report)
                
                return {
                    "message": "Duplicate found! Merged with existing report.",
                    "report_id": existing_report.id,
                    "status": "merged",
                    "priority_band": existing_report.priority_band,
                    "citizen_report_count": existing_report.citizen_report_count
                }

    # LOW RAM MODE: Reduce image size for YOLO processing and force CPU device
    results = model.predict(image, imgsz=320, device='cpu')
    
    # CLEAR MEMORY: Instantly free up RAM after prediction
    gc.collect()

    CONFIDENCE_THRESHOLD = 0.40 
    
    new_hash = str(_compute_phash(image_data))
    
    new_report = Report(status="Reported", image_hash=new_hash)
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    
    # Save image to disk & Log Initial Status
    image_path = f"static/uploads/{new_report.id}.jpg"
    image.save(image_path)

    init_status = StatusHistory(report_id=new_report.id, status="Reported", changed_by="System API")
    db.add(init_status)

    if lat is not None and lon is not None:
        point_geom = f"SRID=4326;POINT({lon} {lat})"
        new_location = Location(
            report_id=new_report.id,
            lat=lat,
            lon=lon,
            source="live_browser_gps",
            geom=point_geom
        )
        db.add(new_location)

    detections_list = []
    max_severity = 0.0     
    max_confidence = 0.0   
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            
            if confidence >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                damage_ratio = box_area / img_area 

                if class_name == "Pothole":
                    severity = min(100.0, (damage_ratio / 0.15) * 100)
                else:
                    severity = min(100.0, (damage_ratio / 0.30) * 100)
                
                final_severity_score = round(severity, 2)

                if final_severity_score > max_severity: 
                    max_severity = final_severity_score
                if confidence > max_confidence: 
                    max_confidence = confidence

                new_detection = Detection(
                    report_id=new_report.id,
                    damage_class=class_name,
                    bbox=[round(x1), round(y1), round(x2), round(y2)],
                    confidence=round(confidence, 2),
                    severity_score=final_severity_score
                )
                db.add(new_detection)
                
                detections_list.append({
                    "damage_type": class_name,
                    "confidence_score": round(confidence, 2),
                    "bounding_box": [round(x1), round(y1), round(x2), round(y2)],
                    "severity_score": final_severity_score
                })

    if len(detections_list) > 0:
        priority_breakdown = compute_priority(
            severity_score=max_severity,
            road_type="unknown", 
            near_sensitive_site=False, 
            citizen_report_count=new_report.citizen_report_count,
            days_unresolved=0,
            detection_confidence=max_confidence
        )
        new_report.priority_score = priority_breakdown.total_score
        new_report.priority_band = priority_breakdown.band
        db.add(new_report)

    db.commit() 
    db.refresh(new_report)

    return {
        "report_id": new_report.id,
        "filename": file.filename,
        "status": "new",
        "priority_band": new_report.priority_band,
        "priority_score": new_report.priority_score,
        "total_issues_found": len(detections_list),
        "detections": detections_list
    }

class StatusUpdate(BaseModel):
    status: str

@app.get("/reports")
def get_reports(db: Session = Depends(get_db)):
    reports = db.query(Report).all()
    result = []
    
    for r in reports:
        loc = r.location
        lat = loc.lat if loc else 0.0
        lon = loc.lon if loc else 0.0
        
        damage_type = "Unknown"
        if r.detections:
            damage_type = r.detections[0].damage_class

        dets = [{"damage_type": d.damage_class, "bbox": d.bbox} for d in r.detections]

        result.append({
            "id": r.id,
            "lat": lat,
            "lon": lon,
            "damage_type": damage_type,
            "status": r.status,
            "priority_score": r.priority_score or 0.0,
            "priority_band": r.priority_band or "LOW",
            "citizen_report_count": r.citizen_report_count,
            "days_unresolved": r.days_unresolved,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "image_url": f"/static/uploads/{r.id}.jpg",
            "ward": "Ward 4" if (r.id % 2 == 0) else "Ward 7", 
            "detections": dets
        })
        
    return result

@app.patch("/reports/{report_id}/status")
def update_report_status(report_id: int, status_update: StatusUpdate, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    report.status = status_update.status
    
    new_history = StatusHistory(report_id=report.id, status=status_update.status, changed_by="Municipal Admin")
    db.add(new_history)
    
    db.commit()
    return {"message": "Status updated successfully", "new_status": report.status}

@app.get("/reports/{report_id}/history")
def get_report_history(report_id: int, db: Session = Depends(get_db)):
    histories = db.query(StatusHistory).filter(StatusHistory.report_id == report_id).order_by(StatusHistory.changed_at.desc()).all()
    return [
        {
            "status": h.status, 
            "changed_at": h.changed_at.isoformat(), 
            "changed_by": h.changed_by
        } 
        for h in histories
    ]

@app.post("/verify-repair")
async def verify_repair(
    report_id: int = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Report Check
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # 2. Anti-Fraud Location Check (PostGIS)
    loc = report.location
    if loc:
        query = text("""
            SELECT ST_DistanceSphere(
                ST_GeomFromText(:point1, 4326),
                ST_GeomFromText(:point2, 4326)
            )
        """)
        distance = db.execute(query, {
            "point1": f"POINT({loc.lon} {loc.lat})",
            "point2": f"POINT({lon} {lat})"
        }).scalar()

        # GPS Tolarance increased to 5000 meters for local testing/demos
        if distance > 5000:
            return {
                "status": "rejected", 
                "reason": f"GPS Fraud Detected: You are {round(distance)} meters away from the actual reported site."
            }

    # 3. Read and Save the 'After' Image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    after_image_path = f"static/uploads/{report_id}_after.jpg"
    image.save(after_image_path)

    # 4. Anti-Fraud AI Check (YOLO) - LOW RAM MODE APPLIED HERE TOO
    results = model.predict(image, imgsz=320, device='cpu')
    gc.collect()

    issues_found = []
    
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= 0.40:
                class_name = model.names[int(box.cls[0])]
                issues_found.append(class_name)

    if len(issues_found) > 0:
        new_history = StatusHistory(report_id=report.id, status="Repair Rejected", changed_by="AI Auditor")
        db.add(new_history)
        db.commit()
        
        return {
            "status": "rejected", 
            "reason": f"AI Quality Check Failed: System still detected {', '.join(issues_found)} in the photo. Please complete the repair properly."
        }

    # 5. Success! Approve and Close Ticket
    report.status = "Verified Closed"
    new_history = StatusHistory(report_id=report.id, status="Verified Closed", changed_by="AI Auditor")
    db.add(new_history)
    db.commit()

    return {
        "status": "approved", 
        "reason": "AI verified smooth asphalt and correct GPS location. Invoice Approved!"
    }

@app.post("/run-sla-audit")
def run_sla_audit(db: Session = Depends(get_db)):
    # Demo purpose ke liye hum un sabhi "CRITICAL" reports ko escalate karenge 
    # jo abhi tak "Verified Closed" ya "Resolved" nahi hui hain.
    open_criticals = db.query(Report).filter(
        Report.priority_band == "CRITICAL",
        Report.status.notin_(["Verified Closed", "Resolved", "ESCALATED"])
    ).all()
    
    escalated_ids = []
    for r in open_criticals:
        r.status = "ESCALATED"
        # Audit Trail mein evidence chhodenge
        history = StatusHistory(
            report_id=r.id, 
            status="ESCALATED", 
            changed_by="Auto-SLA Bot"
        )
        db.add(history)
        escalated_ids.append(r.id)
        
    db.commit()
    
    if len(escalated_ids) > 0:
        return {
            "status": "breach_detected",
            "escalated_count": len(escalated_ids),
            "mock_alert": f"🚨 WhatsApp & Email sent to Municipal Commissioner! {len(escalated_ids)} CRITICAL tasks breached SLA deadlines."
        }
    else:
        return {
            "status": "all_clear",
            "mock_alert": "✅ All critical tasks are within SLA deadlines."
        }