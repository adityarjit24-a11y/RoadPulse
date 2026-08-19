from dataclasses import dataclass
from typing import Optional
import io
import math

import imagehash
from PIL import Image
from sqlalchemy import text
from sqlalchemy.orm import Session

RADIUS_METERS = 25          
PHASH_MAX_DISTANCE = 16 # Increased slightly to compensate for removing heavy AI embeddings

@dataclass
class DuplicateCheckResult:
    is_duplicate: bool
    matched_report_id: Optional[int]
    reason: str 

def _compute_phash(image_bytes: bytes) -> imagehash.ImageHash:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return imagehash.phash(img)

def _calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates distance in meters between two lat/lon points using Haversine formula."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def _get_candidate_reports(db: Session, lat: float, lon: float, radius_m: int = RADIUS_METERS):
    query = text("""
        SELECT r.id AS report_id, r.image_hash, r.status, l.lat, l.lon
        FROM reports r
        JOIN locations l ON l.report_id = r.id
        WHERE r.status NOT IN ('Resolved', 'Verified Closed')
    """)
    
    result = db.execute(query).fetchall()
    
    candidates = []
    for row in result:
        r_id = row[0]
        img_hash = row[1]
        status = row[2]
        r_lat = row[3]
        r_lon = row[4]
        
        if r_lat is not None and r_lon is not None:
            dist = _calculate_haversine_distance(lat, lon, r_lat, r_lon)
            if dist <= radius_m:
                candidates.append({
                    "report_id": r_id,
                    "image_hash": img_hash,
                    "status": status,
                    "distance": dist
                })
                
    candidates.sort(key=lambda x: x["distance"])
    return [(c["report_id"], c["image_hash"], c["status"]) for c in candidates]

def check_for_duplicate(
    db: Session,
    new_image_bytes: bytes,
    lat: float,
    lon: float,
    candidate_image_loader,
) -> DuplicateCheckResult:
    
    candidates = _get_candidate_reports(db, lat, lon)
    if not candidates:
        return DuplicateCheckResult(False, None, "no_candidates")

    new_hash = _compute_phash(new_image_bytes)

    best_match_id = None
    best_distance = None

    for row in candidates:
        if not row[1]: 
            continue
            
        candidate_hash = imagehash.hex_to_hash(row[1])
        distance = new_hash - candidate_hash  
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_match_id = row[0] 

    if best_distance is not None and best_distance <= PHASH_MAX_DISTANCE:
        return DuplicateCheckResult(True, best_match_id, "gps+phash")

    return DuplicateCheckResult(False, None, "no_match")