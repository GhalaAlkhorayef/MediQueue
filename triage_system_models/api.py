from fastapi import FastAPI, HTTPException, Depends, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import uuid
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from contextlib import contextmanager

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DATABASE_URL = "mysql+pymysql://root:@localhost:3307/hospital_triage"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ============================================================================
# DATABASE MODELS
# ============================================================================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password = Column(String(64), nullable=False)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    role = Column(String(20), default="patient", nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    patients = relationship("Patient", back_populates="creator", foreign_keys="Patient.created_by")
    profile = relationship("UserProfile", back_populates="user", uselist=False)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), ForeignKey("users.username"), unique=True, nullable=False)
    
    # Personal Information
    national_id = Column(String(20), unique=True, nullable=True)  # Added for patient identification
    phone = Column(String(20), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    address = Column(Text, nullable=True)
    gender = Column(String(10), nullable=True)
    emergency_contact_name = Column(String(100), nullable=True)
    emergency_contact_phone = Column(String(20), nullable=True)
    emergency_contact_relationship = Column(String(50), nullable=True)
    
    # Medical Information (Removed blood_group as per requirement)
    allergies = Column(Text, nullable=True)
    chronic_conditions = Column(Text, nullable=True)
    current_medications = Column(Text, nullable=True)
    previous_surgeries = Column(Text, nullable=True)
    disabilities = Column(Text, nullable=True)
    
    # Insurance Information
    insurance_provider = Column(String(100), nullable=True)
    insurance_policy_number = Column(String(50), nullable=True)
    
    # Preferences
    preferred_language = Column(String(20), default="English", nullable=True)
    notification_preferences = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    
    user = relationship("User", back_populates="profile")

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    token = Column(String(36), unique=True, index=True, nullable=False)
    username = Column(String(50), nullable=False)
    role = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    expires_at = Column(DateTime, nullable=False)

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(String(20), unique=True, index=True, nullable=False)
    username = Column(String(50), nullable=False)  # Patient's username
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    heart_rate = Column(Integer, nullable=False)
    blood_pressure_systolic = Column(Integer, nullable=False)
    blood_pressure_diastolic = Column(Integer, nullable=False)
    temperature = Column(Float, nullable=False)
    respiratory_rate = Column(Integer, nullable=False)
    oxygen_saturation = Column(Integer, nullable=False)
    pain_level = Column(Integer, nullable=False)
    consciousness_level = Column(String(20), nullable=False)
    patient_complaint = Column(Text, nullable=False)  # Changed from chief_complaint to patient_complaint
    arrival_mode = Column(String(20), nullable=False)
    severity = Column(String(20), nullable=False)
    severity_code = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
    status = Column(String(20), default="Waiting", nullable=False)
    doctor_notes = Column(Text, default="", nullable=True)
    created_by = Column(String(50), ForeignKey("users.username"), nullable=False)  # Nurse who created
    updated_by = Column(String(50), nullable=True)  # Doctor who updated
    last_updated = Column(DateTime, nullable=True)
    estimated_wait_time = Column(Integer, nullable=True)  # Wait time in minutes
    
    creator = relationship("User", back_populates="patients", foreign_keys=[created_by])

Base.metadata.create_all(bind=engine)


class StaffRegistrationRequest(BaseModel):
    username: str = Field(..., min_length=6, max_length=6)  # Employee ID (6 digits)
    password: str = Field(..., min_length=6)
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    role: str = Field(..., pattern="^(admin|doctor|nurse)$")
    
    @validator('username')
    def validate_employee_id(cls, v, values):
        if not v.isdigit():
            raise ValueError('Employee ID must contain only digits')
        if len(v) != 6:
            raise ValueError('Employee ID must be exactly 6 digits')
        
        # Check role-based prefix
        if 'role' in values:
            role = values['role']
            if role == 'admin' and not v.startswith('100'):
                raise ValueError('Admin Employee ID must start with 100')
            elif role == 'doctor' and not v.startswith('200'):
                raise ValueError('Doctor Employee ID must start with 200')
            elif role == 'nurse' and not v.startswith('300'):
                raise ValueError('Nurse Employee ID must start with 300')
        return v

class PatientRegistrationRequest(BaseModel):
    # For nurse to register patients
    national_id: str = Field(..., min_length=10, max_length=20)
    first_name: str = Field(..., min_length=2, max_length=50)
    last_name: str = Field(..., min_length=2, max_length=50)
    date_of_birth: str
    email: EmailStr
    phone: str = Field(..., min_length=10, max_length=20)
    gender: str = Field(..., pattern="^(Male|Female|Other)$")
    address: str
    emergency_contact_name: str
    emergency_contact_phone: str
    emergency_contact_relationship: str
    
    @validator('national_id')
    def validate_national_id(cls, v):
        if not v.isdigit():
            raise ValueError('National ID must contain only digits')
        if len(v) != 10:
            raise ValueError('National ID must be exactly 10 digits')
        return v

class SetPasswordRequest(BaseModel):
    national_id: str = Field(..., min_length=10, max_length=10)
    password: str = Field(..., min_length=6)

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="AI Hospital Triage API",
    description="Emergency Department Triage System API with Role-Based Access Control",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8080",
        "*"  # Allow all origins (for development only!)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================================
# LOAD ML MODEL ARTIFACTS
# ============================================================================

try:
    model = joblib.load('triage_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    severity_mapping = joblib.load('severity_mapping.pkl')
    print("✅ Model artifacts loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading model artifacts: {e}")
    model = None

# ============================================================================
# DATABASE DEPENDENCY
# ============================================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class LoginRequest(BaseModel):
    username: str  # National ID for patients, Employee ID for staff
    password: str

class LoginResponse(BaseModel):
    success: bool
    message: str
    user: Optional[dict] = None
    token: Optional[str] = None
    first_login: Optional[bool] = False  # To indicate if user needs to set password

class UserResponse(BaseModel):
    username: str
    name: str
    email: str
    role: str
    created_at: str
    last_login: Optional[str] = None

class UserProfileRequest(BaseModel):
    phone: Optional[str] = None
    date_of_birth: Optional[str] = None
    address: Optional[str] = None
    gender: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    emergency_contact_relationship: Optional[str] = None
    allergies: Optional[str] = None
    chronic_conditions: Optional[str] = None
    current_medications: Optional[str] = None
    previous_surgeries: Optional[str] = None
    disabilities: Optional[str] = None
    insurance_provider: Optional[str] = None
    insurance_policy_number: Optional[str] = None
    preferred_language: Optional[str] = "English"
    notification_preferences: Optional[str] = None

class UserProfileResponse(BaseModel):
    username: str
    national_id: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[str] = None
    address: Optional[str] = None
    gender: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    emergency_contact_relationship: Optional[str] = None
    allergies: Optional[str] = None
    chronic_conditions: Optional[str] = None
    current_medications: Optional[str] = None
    previous_surgeries: Optional[str] = None
    disabilities: Optional[str] = None
    insurance_provider: Optional[str] = None
    insurance_policy_number: Optional[str] = None
    preferred_language: Optional[str] = None
    notification_preferences: Optional[str] = None
    created_at: str
    updated_at: str

class PatientInput(BaseModel):
    # Nurse inputs patient's national ID to perform triage
    patient_national_id: str = Field(..., min_length=10, max_length=10)
    heart_rate: int = Field(..., ge=40, le=200)
    blood_pressure_systolic: int = Field(..., ge=60, le=250)
    blood_pressure_diastolic: int = Field(..., ge=40, le=150)
    temperature: float = Field(..., ge=35.0, le=42.0)
    respiratory_rate: int = Field(..., ge=8, le=40)
    oxygen_saturation: int = Field(..., ge=70, le=100)
    pain_level: int = Field(..., ge=0, le=10)
    consciousness_level: str = Field(..., pattern="^(Alert|Drowsy|Unresponsive)$")
    patient_complaint: str  # Changed from chief_complaint to patient_complaint
    arrival_mode: str = Field(..., pattern="^(Walk-in|Ambulance|Police)$")

class TriageResponse(BaseModel):
    patient_id: str
    patient_name: str
    severity: str
    severity_code: int
    confidence: float
    timestamp: str
    recommendations: str
    wait_time_estimate: str
    estimated_wait_time_minutes: int

class PatientRecord(BaseModel):
    patient_id: str
    age: int
    gender: str
    patient_complaint: str  # Changed from chief_complaint to patient_complaint
    severity: str
    severity_code: int
    confidence: float
    timestamp: str
    status: str
    doctor_notes: Optional[str] = ""
    estimated_wait_time: Optional[int] = None  # Wait time in minutes
    nurse_name: Optional[str] = None  # Added nurse name

    class Config:
        from_attributes = True

class CompleteUserDataResponse(BaseModel):
    user_info: dict
    profile_info: Optional[dict] = None
    triage_history: List[dict] = []
    total_triages: int = 0
    last_triage: Optional[str] = None

# ============================================================================
# AUTHENTICATION HELPER FUNCTIONS
# ============================================================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_session_token() -> str:
    return str(uuid.uuid4())

def verify_session(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    session = db.query(Session).filter(Session.token == token).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    # Check if session has expired
    if datetime.now() > session.expires_at:
        db.delete(session)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired"
        )
    
    return session.username

def get_current_user(username: str = Depends(verify_session), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user

# ============================================================================
# ML PREDICTION FUNCTIONS
# ============================================================================

def predict_severity(patient_data: dict):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    input_df = pd.DataFrame([patient_data])
    
    for col, le in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col + '_encoded'] = le.transform(input_df[col])
            except:
                input_df[col + '_encoded'] = 0
    
    X = input_df[feature_columns]
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    
    severity_labels = {
        1: 'Non-Urgent',
        2: 'Semi-Urgent',
        3: 'Urgent',
        4: 'Emergency',
        5: 'Critical'
    }
    
    return severity_labels[prediction], proba, int(prediction)

def get_recommendations(severity: str) -> dict:
    recommendations = {
        'Critical': {
            'message': "🚨 IMMEDIATE MEDICAL ATTENTION REQUIRED - Please go to the emergency desk immediately.",
            'wait_time': "Immediate",
            'wait_minutes': 0
        },
        'Emergency': {
            'message': "⚠️ URGENT ATTENTION NEEDED - Please inform the staff at the front desk.",
            'wait_time': "< 15 minutes",
            'wait_minutes': 15
        },
        'Urgent': {
            'message': "🔔 PROMPT ATTENTION RECOMMENDED - Please register at the front desk.",
            'wait_time': "30-60 minutes",
            'wait_minutes': 45
        },
        'Semi-Urgent': {
            'message': "ℹ️ STANDARD PRIORITY - Please register and wait in the designated area.",
            'wait_time': "1-2 hours",
            'wait_minutes': 90
        },
        'Non-Urgent': {
            'message': "✅ ROUTINE CARE - You may be seen by a general practitioner.",
            'wait_time': "2-4 hours",
            'wait_minutes': 180
        }
    }
    return recommendations.get(severity, {'message': 'Please consult with medical staff.', 'wait_time': 'Unknown', 'wait_minutes': 240})

def calculate_wait_time(severity_code: int, db: Session) -> int:
    """Calculate estimated wait time based on severity and current queue"""
    # Get all patients waiting with higher or equal priority
    higher_priority_patients = db.query(Patient).filter(
        Patient.status == "Waiting",
        Patient.severity_code >= severity_code
    ).count()
    
    # Base wait time per severity (in minutes)
    base_wait_times = {
        5: 0,    # Critical
        4: 15,   # Emergency
        3: 45,   # Urgent
        2: 90,   # Semi-Urgent
        1: 180   # Non-Urgent
    }
    
    # Calculate wait time based on queue position
    if severity_code in base_wait_times:
        return base_wait_times[severity_code] + (higher_priority_patients * 5)
    
    return 240  # Default to 4 hours

# ============================================================================
# API ENDPOINTS - PUBLIC
# ============================================================================

@app.get("/")
def read_root():
    return {
        "message": "AI Hospital Triage API with Role-Based Access Control",
        "version": "3.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "database": "MySQL"
    }

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not loaded",
        "database_status": db_status,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# API ENDPOINTS - AUTHENTICATION
# ============================================================================

@app.post("/api/auth/set-password", response_model=LoginResponse, status_code=status.HTTP_201_CREATED)
def set_password(request: SetPasswordRequest, db: Session = Depends(get_db)):
    """Set password for a patient registered by a nurse"""
    try:
        # Check if national ID exists
        profile = db.query(UserProfile).filter(
            UserProfile.national_id == request.national_id
        ).first()
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="National ID not found. Please register first."
            )
        
        # Get user
        user = db.query(User).filter(User.username == profile.username).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Check if password is already set
        if user.password != "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password already set. Please login instead."
            )
        
        # Set password
        user.password = hash_password(request.password)
        db.commit()
        
        # Create session token
        session_token = create_session_token()
        new_session = Session(
            token=session_token,
            username=user.username,
            role=user.role,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
        db.add(new_session)
        db.commit()
        
        return LoginResponse(
            success=True,
            message="Password set successfully",
            user={
                "username": user.username,
                "name": user.name,
                "email": user.email,
                "role": user.role
            },
            token=session_token
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error in set_password: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/login", response_model=LoginResponse)
def login(credentials: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate user and return session token
    - Patients login with National ID (10 digits)
    - Staff (Doctor/Nurse/Admin) login with Employee ID (6 digits)
    """
    try:
        user = db.query(User).filter(User.username == credentials.username).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Check if password is set for patients
        if user.role == "patient" and user.password == "":
            return LoginResponse(
                success=False,
                message="Password not set. Please set your password first.",
                first_login=True
            )
        
        if user.password != hash_password(credentials.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Update last login
        user.last_login = datetime.now()
        db.commit()
        
        # Create session token
        session_token = create_session_token()
        new_session = Session(
            token=session_token,
            username=credentials.username,
            role=user.role,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
        db.add(new_session)
        db.commit()
        
        return LoginResponse(
            success=True,
            message="Login successful",
            user={
                "username": user.username,
                "name": user.name,
                "email": user.email,
                "role": user.role
            },
            token=session_token
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error in login: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    return UserResponse(
        username=current_user.username,
        name=current_user.name,
        email=current_user.email,
        role=current_user.role,
        created_at=current_user.created_at.isoformat(),
        last_login=current_user.last_login.isoformat() if current_user.last_login else None
    )

@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    session = db.query(Session).filter(Session.token == token).first()
    if session:
        db.delete(session)
        db.commit()
    
    return {
        "success": True,
        "message": "Logged out successfully"
    }

# ============================================================================
# API ENDPOINTS - USER PROFILE (PROTECTED)
# ============================================================================

@app.get("/api/user/profile", response_model=UserProfileResponse)
def get_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user profile information - Patients can only see their own"""
    try:
        profile = db.query(UserProfile).filter(UserProfile.username == current_user.username).first()
        
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        return UserProfileResponse(
            username=profile.username,
            national_id=profile.national_id,
            phone=profile.phone,
            date_of_birth=profile.date_of_birth.isoformat() if profile.date_of_birth else None,
            address=profile.address,
            gender=profile.gender,
            emergency_contact_name=profile.emergency_contact_name,
            emergency_contact_phone=profile.emergency_contact_phone,
            emergency_contact_relationship=profile.emergency_contact_relationship,
            allergies=profile.allergies,
            chronic_conditions=profile.chronic_conditions,
            current_medications=profile.current_medications,
            previous_surgeries=profile.previous_surgeries,
            disabilities=profile.disabilities,
            insurance_provider=profile.insurance_provider,
            insurance_policy_number=profile.insurance_policy_number,
            preferred_language=profile.preferred_language,
            notification_preferences=profile.notification_preferences,
            created_at=profile.created_at.isoformat(),
            updated_at=profile.updated_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_user_profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/user/profile")
def update_user_profile(
    profile_data: UserProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile information"""
    try:
        profile = db.query(UserProfile).filter(UserProfile.username == current_user.username).first()
        
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        update_data = profile_data.dict(exclude_unset=True)
        
        if 'date_of_birth' in update_data and update_data['date_of_birth']:
            try:
                update_data['date_of_birth'] = datetime.fromisoformat(update_data['date_of_birth'].replace('Z', '+00:00'))
            except:
                update_data['date_of_birth'] = None
        
        for key, value in update_data.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.updated_at = datetime.now()
        db.commit()
        db.refresh(profile)
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "profile": {
                "username": profile.username,
                "updated_at": profile.updated_at.isoformat()
            }
        }
    except Exception as e:
        db.rollback()
        print(f"Error in update_user_profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/complete-data", response_model=CompleteUserDataResponse)
def get_complete_user_data(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get complete user data - Patients can only see their own data"""
    try:
        user_info = {
            "username": current_user.username,
            "name": current_user.name,
            "email": current_user.email,
            "role": current_user.role,
            "created_at": current_user.created_at.isoformat(),
            "last_login": current_user.last_login.isoformat() if current_user.last_login else None
        }
        
        profile = db.query(UserProfile).filter(UserProfile.username == current_user.username).first()
        profile_info = None
        
        if profile:
            profile_info = {
                "national_id": profile.national_id,
                "phone": profile.phone,
                "date_of_birth": profile.date_of_birth.isoformat() if profile.date_of_birth else None,
                "address": profile.address,
                "gender": profile.gender,
                "emergency_contact_name": profile.emergency_contact_name,
                "emergency_contact_phone": profile.emergency_contact_phone,
                "emergency_contact_relationship": profile.emergency_contact_relationship,
                "allergies": profile.allergies,
                "chronic_conditions": profile.chronic_conditions,
                "current_medications": profile.current_medications,
                "previous_surgeries": profile.previous_surgeries,
                "disabilities": profile.disabilities,
                "insurance_provider": profile.insurance_provider,
                "insurance_policy_number": profile.insurance_policy_number,
                "preferred_language": profile.preferred_language,
                "notification_preferences": profile.notification_preferences,
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat()
            }
        
        # Get triage history - only for the current user
        triage_records = db.query(Patient).filter(
            Patient.username == current_user.username
        ).order_by(Patient.timestamp.desc()).all()
        
        triage_history = []
        for record in triage_records:
            # Get nurse name
            nurse = db.query(User).filter(User.username == record.created_by).first()
            nurse_name = nurse.name if nurse else "Unknown"
            
            triage_history.append({
                "patient_id": record.patient_id,
                "age": record.age,
                "gender": record.gender,
                "heart_rate": record.heart_rate,
                "blood_pressure_systolic": record.blood_pressure_systolic,
                "blood_pressure_diastolic": record.blood_pressure_diastolic,
                "temperature": record.temperature,
                "respiratory_rate": record.respiratory_rate,
                "oxygen_saturation": record.oxygen_saturation,
                "pain_level": record.pain_level,
                "consciousness_level": record.consciousness_level,
                "patient_complaint": record.patient_complaint,  # Changed from chief_complaint
                "arrival_mode": record.arrival_mode,
                "severity": record.severity,
                "severity_code": record.severity_code,
                "confidence": record.confidence,
                "timestamp": record.timestamp.isoformat(),
                "status": record.status,
                "doctor_notes": record.doctor_notes or "",
                "created_by": record.created_by,
                "nurse_name": nurse_name,  # Added nurse name
                "estimated_wait_time": record.estimated_wait_time  # Added wait time
            })
        
        return CompleteUserDataResponse(
            user_info=user_info,
            profile_info=profile_info,
            triage_history=triage_history,
            total_triages=len(triage_history),
            last_triage=triage_history[0]["timestamp"] if triage_history else None
        )
    except Exception as e:
        print(f"Error in get_complete_user_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# API ENDPOINTS - TRIAGE (ROLE-BASED ACCESS)
# ============================================================================

@app.post("/api/nurse/register-patient", status_code=status.HTTP_201_CREATED)
def register_patient(
    patient_data: PatientRegistrationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Register a new patient - Only nurses can register patients
    Patient will need to set password on first login
    """
    try:
        # Check if user is a nurse
        if current_user.role != "nurse":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Only nurses can register patients."
            )
        
        # Check if national ID already exists
        existing_profile = db.query(UserProfile).filter(
            UserProfile.national_id == patient_data.national_id
        ).first()
        if existing_profile:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="National ID already registered"
            )
        
        # Check if email already exists
        existing_email = db.query(User).filter(User.email == patient_data.email).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Use national ID as username for patients
        username = patient_data.national_id
        full_name = f"{patient_data.first_name} {patient_data.last_name}"
        
        # Create new user with empty password (to be set by patient)
        new_user = User(
            username=username,
            password="",  # Empty password for patient to set
            name=full_name,
            email=patient_data.email,
            role="patient",
            created_at=datetime.now()
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Create profile with all details
        try:
            dob = datetime.fromisoformat(patient_data.date_of_birth.replace('Z', '+00:00'))
        except:
            dob = None
        
        new_profile = UserProfile(
            username=username,
            national_id=patient_data.national_id,
            phone=patient_data.phone,
            date_of_birth=dob,
            address=patient_data.address,
            gender=patient_data.gender,
            emergency_contact_name=patient_data.emergency_contact_name,
            emergency_contact_phone=patient_data.emergency_contact_phone,
            emergency_contact_relationship=patient_data.emergency_contact_relationship,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        db.add(new_profile)
        db.commit()
        
        return {
            "success": True,
            "message": "Patient registered successfully. Patient can now login with national ID and set password.",
            "patient": {
                "username": username,
                "name": full_name,
                "email": patient_data.email,
                "role": "patient"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error in register_patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/triage/predict", response_model=TriageResponse)
def predict_triage(
    patient: PatientInput,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Predict triage severity for a patient
    ONLY NURSES can perform triage tests
    The result will appear on the patient's dashboard
    """
    try:
        # Check if user is a nurse
        if current_user.role != "nurse":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Only nurses can perform triage tests."
            )
        
        # Find patient by national ID
        patient_profile = db.query(UserProfile).filter(
            UserProfile.national_id == patient.patient_national_id
        ).first()
        
        if not patient_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient with this National ID not found. Please ensure the patient is registered."
            )
        
        # Get patient user details
        patient_user = db.query(User).filter(
            User.username == patient_profile.username
        ).first()
        
        if not patient_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient user record not found"
            )
        
        # Calculate age from date of birth
        age = 0
        if patient_profile.date_of_birth:
            age = (datetime.now() - patient_profile.date_of_birth).days // 365
        
        # Get gender from profile
        gender = patient_profile.gender or "Male"
        
        # Prepare data for prediction
        patient_data = {
            'age': age,
            'gender': gender,
            'heart_rate': patient.heart_rate,
            'blood_pressure_systolic': patient.blood_pressure_systolic,
            'blood_pressure_diastolic': patient.blood_pressure_diastolic,
            'temperature': patient.temperature,
            'respiratory_rate': patient.respiratory_rate,
            'oxygen_saturation': patient.oxygen_saturation,
            'pain_level': patient.pain_level,
            'consciousness_level': patient.consciousness_level,
            'chief_complaint': patient.patient_complaint,  # Changed field name but keep for model compatibility
            'arrival_mode': patient.arrival_mode
        }
        
        # Make prediction
        severity, proba, severity_code = predict_severity(patient_data)
        
        # Generate patient ID
        patient_count = db.query(Patient).count()
        patient_id = f"P{patient_count + 1:05d}"
        
        # Get recommendations and calculate wait time
        rec = get_recommendations(severity)
        estimated_wait_time = calculate_wait_time(severity_code, db)
        
        # Create patient record linked to the patient's username
        new_patient = Patient(
            patient_id=patient_id,
            username=patient_profile.username,  # Link to patient's account
            age=age,
            gender=gender,
            heart_rate=patient.heart_rate,
            blood_pressure_systolic=patient.blood_pressure_systolic,
            blood_pressure_diastolic=patient.blood_pressure_diastolic,
            temperature=patient.temperature,
            respiratory_rate=patient.respiratory_rate,
            oxygen_saturation=patient.oxygen_saturation,
            pain_level=patient.pain_level,
            consciousness_level=patient.consciousness_level,
            patient_complaint=patient.patient_complaint,  # Changed from chief_complaint
            arrival_mode=patient.arrival_mode,
            severity=severity,
            severity_code=severity_code,
            confidence=float(max(proba) * 100),
            timestamp=datetime.now(),
            status="Waiting",
            created_by=current_user.username,  # Nurse who performed the triage
            estimated_wait_time=estimated_wait_time  # Added wait time
        )
        
        db.add(new_patient)
        db.commit()
        db.refresh(new_patient)
        
        return TriageResponse(
            patient_id=patient_id,
            patient_name=patient_user.name,
            severity=severity,
            severity_code=severity_code,
            confidence=float(max(proba) * 100),
            timestamp=datetime.now().isoformat(),
            recommendations=rec['message'],
            wait_time_estimate=rec['wait_time'],
            estimated_wait_time_minutes=estimated_wait_time  # Added wait time in minutes
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error in predict_triage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/triage/history")
def get_triage_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get triage history
    - PATIENTS: See only their own triage records
    - NURSES: See all triage records (read-only)
    - DOCTORS: See all triage records
    """
    try:
        if current_user.role == "patient":
            # Patients can only see their own records
            records = db.query(Patient).filter(
                Patient.username == current_user.username
            ).order_by(Patient.timestamp.desc()).all()
        else:
            # Nurses and doctors can see all records
            records = db.query(Patient).order_by(
                Patient.severity_code.desc(),
                Patient.timestamp.desc()
            ).all()
        
        # Get nurse names for all records
        nurse_names = {}
        for record in records:
            if record.created_by not in nurse_names:
                nurse = db.query(User).filter(User.username == record.created_by).first()
                nurse_names[record.created_by] = nurse.name if nurse else "Unknown"
        
        return {
            "success": True,
            "total": len(records),
            "records": [
                {
                    "patient_id": r.patient_id,
                    "patient_username": r.username,
                    "age": r.age,
                    "gender": r.gender,
                    "patient_complaint": r.patient_complaint,  # Changed from chief_complaint
                    "severity": r.severity,
                    "severity_code": r.severity_code,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp.isoformat(),
                    "status": r.status,
                    "doctor_notes": r.doctor_notes or "",
                    "created_by": r.created_by,
                    "nurse_name": nurse_names.get(r.created_by, "Unknown"),  # Added nurse name
                    "estimated_wait_time": r.estimated_wait_time,  # Added wait time
                    "vital_signs": {
                        "heart_rate": r.heart_rate,
                        "blood_pressure": f"{r.blood_pressure_systolic}/{r.blood_pressure_diastolic}",
                        "temperature": r.temperature,
                        "respiratory_rate": r.respiratory_rate,
                        "oxygen_saturation": r.oxygen_saturation,
                        "pain_level": r.pain_level,
                        "consciousness_level": r.consciousness_level
                    }
                }
                for r in records
            ]
        }
    except Exception as e:
        print(f"Error in get_triage_history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/triage/patients")
def get_all_patients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    filter_option: Optional[str] = None  # For admin dropdown filter
):
    """
    Get all patients
    - DOCTORS: Can view all patients (most recent records first)
    - NURSES: Can view all patients (read-only)
    - PATIENTS: Cannot access this endpoint
    - ADMIN: Can view all patients with filter options
    """
    try:
        if current_user.role == "patient":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Patients cannot view all patient records."
            )
        
        if current_user.role not in ["doctor", "nurse", "admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Only medical staff can view all patients."
            )
        
        # Base query
        query = db.query(Patient)
        
        # Apply filters for admin
        if current_user.role == "admin" and filter_option:
            if filter_option == "waiting":
                query = query.filter(Patient.status == "Waiting")
            elif filter_option == "in_treatment":
                query = query.filter(Patient.status == "In Treatment")
            elif filter_option == "completed":
                query = query.filter(Patient.status == "Completed")
            elif filter_option == "critical":
                query = query.filter(Patient.severity == "Critical")
            elif filter_option == "emergency":
                query = query.filter(Patient.severity == "Emergency")
            elif filter_option == "today":
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                query = query.filter(Patient.timestamp >= today_start)
        
        # For doctors, show most recent records first
        if current_user.role == "doctor":
            # Use case statement to prioritize records with last_updated not null
            from sqlalchemy import case, desc
            query = query.order_by(
                case(
                    (Patient.last_updated.isnot(None), 0),
                    else_=1
                ),
                desc(Patient.last_updated),
                desc(Patient.timestamp)
            )
        else:
            # For nurses and admins, sort by priority
            query = query.order_by(Patient.severity_code.desc(), Patient.timestamp.asc())
        
        patients = query.all()
        
        patient_list = []
        for p in patients:
            # Get patient's full information
            patient_user = db.query(User).filter(User.username == p.username).first()
            patient_profile = db.query(UserProfile).filter(UserProfile.username == p.username).first()
            
            # Get nurse name
            nurse = db.query(User).filter(User.username == p.created_by).first()
            nurse_name = nurse.name if nurse else "Unknown"
            
            patient_info = {
                "patient_id": p.patient_id,
                "username": p.username,
                "name": patient_user.name if patient_user else "Unknown",
                "national_id": patient_profile.national_id if patient_profile else None,
                "age": p.age,
                "gender": p.gender,
                "patient_complaint": p.patient_complaint,  # Changed from chief_complaint
                "severity": p.severity,
                "severity_code": p.severity_code,
                "confidence": p.confidence,
                "timestamp": p.timestamp.isoformat(),
                "status": p.status,
                "doctor_notes": p.doctor_notes or "",
                "created_by": p.created_by,
                "nurse_name": nurse_name,  # Added nurse name
                "updated_by": p.updated_by,
                "last_updated": p.last_updated.isoformat() if p.last_updated else None,
                "estimated_wait_time": p.estimated_wait_time,  # Added wait time
                "vital_signs": {
                    "heart_rate": p.heart_rate,
                    "blood_pressure": f"{p.blood_pressure_systolic}/{p.blood_pressure_diastolic}",
                    "temperature": p.temperature,
                    "respiratory_rate": p.respiratory_rate,
                    "oxygen_saturation": p.oxygen_saturation,
                    "pain_level": p.pain_level,
                    "consciousness_level": p.consciousness_level
                }
            }
            
            # Add medical history if available
            if patient_profile:
                patient_info["medical_history"] = {
                    "allergies": patient_profile.allergies,
                    "chronic_conditions": patient_profile.chronic_conditions,
                    "current_medications": patient_profile.current_medications,
                    "previous_surgeries": patient_profile.previous_surgeries
                }
            
            patient_list.append(patient_info)
        
        return {
            "success": True,
            "total": len(patient_list),
            "patients": patient_list
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_all_patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/triage/patient/{patient_id}")
def update_patient_status(
    patient_id: str,
    status: str,
    doctor_notes: Optional[str] = "",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update patient status and notes
    ONLY DOCTORS can update patient records
    Nurses cannot update status
    """
    try:
        # Check if user is a doctor
        if current_user.role != "doctor":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Only doctors can update patient status and notes."
            )
        
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Validate status
        valid_statuses = ["Waiting", "In Treatment", "Completed", "Discharged", "Referred"]
        if status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )
        
        patient.status = status
        if doctor_notes:
            if patient.doctor_notes:
                # Append new notes with timestamp
                patient.doctor_notes += f"\n\n[{datetime.now().strftime('%Y-%m-%d %H:%M')} - Dr. {current_user.name}]\n{doctor_notes}"
            else:
                patient.doctor_notes = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')} - Dr. {current_user.name}]\n{doctor_notes}"
        
        patient.updated_by = current_user.username
        patient.last_updated = datetime.now()
        
        # Recalculate wait times for all waiting patients if status changed from "Waiting"
        if patient.status != "Waiting":
            waiting_patients = db.query(Patient).filter(Patient.status == "Waiting").all()
            for p in waiting_patients:
                p.estimated_wait_time = calculate_wait_time(p.severity_code, db)
        
        db.commit()
        
        return {
            "success": True,
            "message": "Patient record updated successfully",
            "patient_id": patient_id,
            "status": status,
            "updated_by": current_user.name,
            "updated_at": patient.last_updated.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error in update_patient_status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/triage/patient/{patient_id}")
def get_patient_details(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed patient information
    - DOCTORS & NURSES: Can view any patient
    - PATIENTS: Can only view their own records
    """
    try:
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Check authorization
        if current_user.role == "patient" and patient.username != current_user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You can only view your own records."
            )
        
        # Get patient user and profile
        patient_user = db.query(User).filter(User.username == patient.username).first()
        patient_profile = db.query(UserProfile).filter(UserProfile.username == patient.username).first()
        
        # Get nurse name
        nurse = db.query(User).filter(User.username == patient.created_by).first()
        nurse_name = nurse.name if nurse else "Unknown"
        
        result = {
            "patient_id": patient.patient_id,
            "name": patient_user.name if patient_user else "Unknown",
            "national_id": patient_profile.national_id if patient_profile else None,
            "age": patient.age,
            "gender": patient.gender,
            "patient_complaint": patient.patient_complaint,  # Changed from chief_complaint
            "arrival_mode": patient.arrival_mode,
            "severity": patient.severity,
            "severity_code": patient.severity_code,
            "confidence": patient.confidence,
            "timestamp": patient.timestamp.isoformat(),
            "status": patient.status,
            "doctor_notes": patient.doctor_notes or "",
            "created_by": patient.created_by,
            "nurse_name": nurse_name,  # Added nurse name
            "updated_by": patient.updated_by,
            "last_updated": patient.last_updated.isoformat() if patient.last_updated else None,
            "estimated_wait_time": patient.estimated_wait_time,  # Added wait time
            "vital_signs": {
                "heart_rate": patient.heart_rate,
                "blood_pressure_systolic": patient.blood_pressure_systolic,
                "blood_pressure_diastolic": patient.blood_pressure_diastolic,
                "temperature": patient.temperature,
                "respiratory_rate": patient.respiratory_rate,
                "oxygen_saturation": patient.oxygen_saturation,
                "pain_level": patient.pain_level,
                "consciousness_level": patient.consciousness_level
            }
        }
        
        # Add medical history for doctors and nurses
        if current_user.role in ["doctor", "nurse", "admin"] and patient_profile:
            result["medical_history"] = {
                "allergies": patient_profile.allergies,
                "chronic_conditions": patient_profile.chronic_conditions,
                "current_medications": patient_profile.current_medications,
                "previous_surgeries": patient_profile.previous_surgeries,
                "disabilities": patient_profile.disabilities
            }
            result["emergency_contact"] = {
                "name": patient_profile.emergency_contact_name,
                "phone": patient_profile.emergency_contact_phone,
                "relationship": patient_profile.emergency_contact_relationship
            }
        
        return {
            "success": True,
            "patient": result
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_patient_details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats/dashboard")
def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get dashboard statistics based on user role"""
    try:
        if current_user.role == "patient":
            # Patient stats - only their own data
            total_visits = db.query(Patient).filter(
                Patient.username == current_user.username
            ).count()
            
            latest_visit = db.query(Patient).filter(
                Patient.username == current_user.username
            ).order_by(Patient.timestamp.desc()).first()
            
            return {
                "success": True,
                "user_role": "patient",
                "stats": {
                    "total_visits": total_visits,
                    "latest_status": latest_visit.status if latest_visit else None,
                    "latest_severity": latest_visit.severity if latest_visit else None,
                    "latest_visit": latest_visit.timestamp.isoformat() if latest_visit else None,
                    "latest_wait_time": latest_visit.estimated_wait_time if latest_visit else None
                }
            }
        else:
            # Medical staff stats
            total_patients = db.query(Patient).count()
            waiting = db.query(Patient).filter(Patient.status == "Waiting").count()
            in_treatment = db.query(Patient).filter(Patient.status == "In Treatment").count()
            completed = db.query(Patient).filter(Patient.status == "Completed").count()
            
            # Severity breakdown
            critical = db.query(Patient).filter(Patient.severity == "Critical", Patient.status == "Waiting").count()
            emergency = db.query(Patient).filter(Patient.severity == "Emergency", Patient.status == "Waiting").count()
            urgent = db.query(Patient).filter(Patient.severity == "Urgent", Patient.status == "Waiting").count()
            
            # Today's stats
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_patients = db.query(Patient).filter(Patient.timestamp >= today_start).count()
            
            # Average wait time
            avg_wait_time = db.query(Patient).filter(
                Patient.status == "Waiting",
                Patient.estimated_wait_time.isnot(None)
            ).with_entities(func.avg(Patient.estimated_wait_time)).scalar()
            
            stats = {
                "total_patients": total_patients,
                "today_patients": today_patients,
                "waiting": waiting,
                "in_treatment": in_treatment,
                "completed": completed,
                "critical_waiting": critical,
                "emergency_waiting": emergency,
                "urgent_waiting": urgent,
                "average_wait_time": int(avg_wait_time) if avg_wait_time else 0
            }
            
            # Add role-specific stats
            if current_user.role == "nurse":
                my_triages = db.query(Patient).filter(
                    Patient.created_by == current_user.username
                ).count()
                stats["my_triages_performed"] = my_triages
            elif current_user.role == "doctor":
                my_patients = db.query(Patient).filter(
                    Patient.updated_by == current_user.username
                ).count()
                stats["my_patients_treated"] = my_patients
            
            return {
                "success": True,
                "user_role": current_user.role,
                "stats": stats
            }
    except Exception as e:
        print(f"Error in get_dashboard_stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search/patient")
def search_patient_by_national_id(
    national_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search for a patient by National ID
    Only NURSES and DOCTORS can search
    Used before performing triage
    """
    try:
        if current_user.role not in ["nurse", "doctor", "admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Only medical staff can search for patients."
            )
        
        # Find patient profile
        patient_profile = db.query(UserProfile).filter(
            UserProfile.national_id == national_id
        ).first()
        
        if not patient_profile:
            return {
                "success": False,
                "message": "Patient not found. Please register the patient first."
            }
        
        # Get patient user details
        patient_user = db.query(User).filter(
            User.username == patient_profile.username
        ).first()
        
        if not patient_user:
            return {
                "success": False,
                "message": "Patient user record not found"
            }
        
        # Get triage history count
        triage_count = db.query(Patient).filter(
            Patient.username == patient_profile.username
        ).count()
        
        # Calculate age
        age = 0
        if patient_profile.date_of_birth:
            age = (datetime.now() - patient_profile.date_of_birth).days // 365
        
        return {
            "success": True,
            "patient": {
                "national_id": patient_profile.national_id,
                "name": patient_user.name,
                "age": age,
                "gender": patient_profile.gender,
                "phone": patient_profile.phone,
                "allergies": patient_profile.allergies,
                "chronic_conditions": patient_profile.chronic_conditions,
                "current_medications": patient_profile.current_medications,
                "previous_visits": triage_count,
                "emergency_contact": {
                    "name": patient_profile.emergency_contact_name,
                    "phone": patient_profile.emergency_contact_phone,
                    "relationship": patient_profile.emergency_contact_relationship
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in search_patient_by_national_id: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# API ENDPOINTS - STAFF REGISTRATION (PUBLIC - FOR ADMIN USE)
# ============================================================================

@app.post("/api/admin/register-staff", status_code=status.HTTP_201_CREATED)
def register_staff(
    staff_data: StaffRegistrationRequest,
    db: Session = Depends(get_db)
):
    """
    Register new staff member (Admin, Doctor, or Nurse)
    This is a public endpoint for initial staff setup
    In production, this should be protected with admin authentication
    """
    try:
        # Check if employee ID already exists
        existing_user = db.query(User).filter(User.username == staff_data.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Employee ID already registered"
            )
        
        # Check if email already exists
        existing_email = db.query(User).filter(User.email == staff_data.email).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new staff user
        hashed_password = hash_password(staff_data.password)
        new_staff = User(
            username=staff_data.username,
            password=hashed_password,
            name=staff_data.name,
            email=staff_data.email,
            role=staff_data.role,
            created_at=datetime.now()
        )
        
        db.add(new_staff)
        db.commit()
        db.refresh(new_staff)
        
        return {
            "success": True,
            "message": f"{staff_data.role.capitalize()} registered successfully",
            "user": {
                "username": new_staff.username,
                "name": new_staff.name,
                "email": new_staff.email,
                "role": new_staff.role,
                "created_at": new_staff.created_at.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error in register_staff: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)