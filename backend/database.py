from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

# YEH LINE BOHOT IMPORTANT HAI, YE SQLITE HONI CHAHIYE POSTGRESQL NAHI
SQLALCHEMY_DATABASE_URL = "sqlite:///./roadpulse_final.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

from sqlalchemy import text

# Auto-fix: Add missing image_hash column if it doesn't exist yet
try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE reports ADD COLUMN image_hash VARCHAR;"))
        conn.commit()
        print("Successfully added missing image_hash column!")
except Exception as e:
    # Column pehle se hoga toh error ko ignore kar dega
    pass

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    image_url = Column(String)
    total_issues_found = Column(Integer)
    priority_score = Column(Float)
    priority_band = Column(String)
    status = Column(String, default="Reported")
    damage_type = Column(String)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    damage_type = Column(String)
    confidence_score = Column(Float)
    bbox = Column(JSON)

class Location(Base):
    __tablename__ = "locations"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    lat = Column(Float)
    lon = Column(Float)

class StatusHistory(Base):
    __tablename__ = "status_history"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    status = Column(String)
    changed_at = Column(DateTime, default=datetime.datetime.utcnow)
    changed_by = Column(String)

Base.metadata.create_all(bind=engine)