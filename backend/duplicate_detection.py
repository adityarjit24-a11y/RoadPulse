from dataclasses import dataclass
from typing import Optional
import io

import imagehash
from PIL import Image
from sqlalchemy import text
from sqlalchemy.orm import Session

RADIUS_METERS = 25          
PHASH_MAX_DISTANCE = 8      
PHASH_AMBIGUOUS_BAND = (9, 16)   
EMBEDDING_SIM_THRESHOLD = 0.90    

@dataclass
class DuplicateCheckResult:
    is_duplicate: bool
    matched_report_id: Optional[int]
    reason: str 

def _compute_phash(image_bytes: bytes) -> imagehash.ImageHash:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return imagehash.phash(img)

def _get_candidate_reports(db: Session, lat: float, lon: float, radius_m: int = RADIUS_METERS):
    query = text("""
        SELECT r.id AS report_id, r.image_hash, r.status
        FROM reports r
        JOIN locations l ON l.report_id = r.id
        WHERE r.status NOT IN ('Resolved', 'Verified Closed')
          AND ST_DWithin(
                l.geom::geography,
                ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                :radius_m
              )
        ORDER BY ST_Distance(
            l.geom::geography,
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
        )
    """)
    return db.execute(query, {"lon": lon, "lat": lat, "radius_m": radius_m}).fetchall()

_embedding_model = None
_embedding_transform = None

def _load_embedding_model():
    global _embedding_model, _embedding_transform
    if _embedding_model is None:
        import torch
        from torchvision import models, transforms

        base = models.mobilenet_v3_small(weights="DEFAULT")
        base.classifier = torch.nn.Identity()  
        base.eval()
        _embedding_model = base
        _embedding_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _embedding_model, _embedding_transform

def _compute_embedding(image_bytes: bytes):
    import torch
    model, transform = _load_embedding_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = model(tensor).squeeze(0)
    return vec

def _cosine_similarity(a, b) -> float:
    import torch
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

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
        # Agar purani report mein hash nahi hai toh skip karo
        if not row.image_hash:
            continue
            
        candidate_hash = imagehash.hex_to_hash(row.image_hash)
        distance = new_hash - candidate_hash  
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_match_id = row.report_id

    if best_distance is not None and best_distance <= PHASH_MAX_DISTANCE:
        return DuplicateCheckResult(True, best_match_id, "gps+phash")

    if best_distance is not None and PHASH_AMBIGUOUS_BAND[0] <= best_distance <= PHASH_AMBIGUOUS_BAND[1]:
        new_embedding = _compute_embedding(new_image_bytes)
        try:
            candidate_bytes = candidate_image_loader(best_match_id)
            candidate_embedding = _compute_embedding(candidate_bytes)
            similarity = _cosine_similarity(new_embedding, candidate_embedding)
            if similarity >= EMBEDDING_SIM_THRESHOLD:
                return DuplicateCheckResult(True, best_match_id, "gps+embedding")
        except Exception as e:
            print(f"Embedding check failed: {e}")

    return DuplicateCheckResult(False, None, "no_match")