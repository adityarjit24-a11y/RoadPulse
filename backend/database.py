import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON, text, inspect
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

SQLALCHEMY_DATABASE_URL = "sqlite:///./roadpulse_v4.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    image_url = Column(String, nullable=True)
    total_issues_found = Column(Integer, default=0)
    priority_score = Column(Float, nullable=True)
    priority_band = Column(String, nullable=True)
    status = Column(String, default="Reported")
    damage_type = Column(String, nullable=True)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Missing Columns Fix
    image_hash = Column(String, nullable=True)
    citizen_report_count = Column(Integer, default=1)
    days_unresolved = Column(Integer, default=0)

    # Relationships
    location = relationship("Location", uselist=False, back_populates="report")
    detections = relationship("Detection", back_populates="report")

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    damage_class = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    severity_score = Column(Float, nullable=True)
    bbox = Column(JSON, nullable=True)

    report = relationship("Report", back_populates="detections")

class Location(Base):
    __tablename__ = "locations"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    source = Column(String, nullable=True)
    geom = Column(String, nullable=True)

    report = relationship("Report", back_populates="location")

class StatusHistory(Base):
    __tablename__ = "status_history"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    status = Column(String)
    changed_at = Column(DateTime, default=datetime.datetime.utcnow)
    changed_by = Column(String)

# Tables Create karo
Base.metadata.create_all(bind=engine)

# Dynamic Schema Migration (Har startup par check karega aur missing columns khud add kar dega)
def auto_migrate():
    inspector = inspect(engine)
    
    def add_col_if_missing(table, column, col_type):
        if table in inspector.get_table_names():
            cols = [c['name'] for c in inspector.get_columns(table)]
            if column not in cols:
                try:
                    with engine.connect() as conn:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type};"))
                        conn.commit()
                        print(f"Auto-Migration: Added '{column}' to '{table}'")
                except Exception as e:
                    pass

    add_col_if_missing('reports', 'image_hash', 'VARCHAR')
    add_col_if_missing('reports', 'citizen_report_count', 'INTEGER DEFAULT 1')
    add_col_if_missing('reports', 'days_unresolved', 'INTEGER DEFAULT 0')
    add_col_if_missing('locations', 'source', 'VARCHAR')
    add_col_if_missing('locations', 'geom', 'VARCHAR')
    add_col_if_missing('detections', 'damage_class', 'VARCHAR')
    add_col_if_missing('detections', 'confidence', 'FLOAT')
    add_col_if_missing('detections', 'severity_score', 'FLOAT')

auto_migrate()