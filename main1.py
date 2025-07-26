# SMART BMI APP - FULL BACKEND COMBINED (FastAPI + JWT + AI/ML + Analytics)

from fastapi import FastAPI, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io, base64, numpy as np
import torch, torch.nn as nn
app = FastAPI()

# -------------------- CONFIG --------------------
SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

DATABASE_URL = "sqlite:///./bmi_app.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- DATABASE --------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    hashed_password = Column(String)
    bmi_logs = relationship("BMILog", back_populates="user")

class BMILog(Base):
    __tablename__ = "bmi_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    weight = Column(Float)
    height = Column(Float)
    bmi = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="bmi_logs")

Base.metadata.create_all(bind=engine)

# -------------------- AI MODEL --------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

ai_model = SimpleNN()
ai_model.load_state_dict(torch.load("bmi_model.pt"))  # Ensure this model file exists
ai_model.eval()

# -------------------- UTILS --------------------
def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user(db, username: str):
    return db.query(User).filter(User.username == username).first()

def get_current_user(db: Session = Depends(SessionLocal), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=username)
    if user is None:
        raise credentials_exception
    return user

# -------------------- AUTH ROUTES --------------------
@app.post("/register")
def register(username: str = Form(...), password: str = Form(...), db: Session = Depends(SessionLocal)):
    if get_user(db, username):
        raise HTTPException(status_code=400, detail="Username already registered")
    user = User(username=username, hashed_password=get_password_hash(password))
    db.add(user)
    db.commit()
    return {"message": "User registered successfully"}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(SessionLocal)):
    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=30))
    return {"access_token": access_token, "token_type": "bearer"}

# -------------------- BMI ROUTES --------------------
@app.post("/bmi")
def calculate_bmi(weight: float, height: float, db: Session = Depends(SessionLocal), current_user: User = Depends(get_current_user)):
    height_m = height / 100
    bmi = round(weight / (height_m ** 2), 2)
    log = BMILog(user_id=current_user.id, weight=weight, height=height, bmi=bmi)
    db.add(log)
    db.commit()
    return {"bmi": bmi, "message": "BMI calculated and saved"}

@app.get("/bmi/history")
def bmi_history(db: Session = Depends(SessionLocal), current_user: User = Depends(get_current_user)):
    logs = db.query(BMILog).filter(BMILog.user_id == current_user.id).order_by(BMILog.timestamp).all()
    return [{"date": log.timestamp.strftime("%Y-%m-%d"), "bmi": log.bmi} for log in logs]

# -------------------- AI + DIET --------------------
@app.post("/predict_bmi_ai")
def predict_bmi_ai(age: int, weight: float, height: float):
    input_tensor = torch.tensor([[age, weight, height]], dtype=torch.float32)
    with torch.no_grad():
        pred = ai_model(input_tensor).item()
    bmi = round(pred, 2)
    if bmi < 18:
        diet = "Increase calorie intake: nuts, dairy, rice, peanut butter."
    elif 18 <= bmi <= 24.9:
        diet = "Balanced diet: whole grains, lean protein, fruits, veggies."
    else:
        diet = "Low-fat, high-fiber foods: oatmeal, beans, veggies, low-fat yogurt."
    return {"predicted_bmi": bmi, "diet_recommendation": diet}

# -------------------- DASHBOARD --------------------
@app.get("/bmi/dashboard")
def bmi_dashboard(current_user: User = Depends(get_current_user), db: Session = Depends(SessionLocal)):
    logs = db.query(BMILog).filter(BMILog.user_id == current_user.id).order_by(BMILog.timestamp).all()
    if not logs:
        return {"message": "No BMI records found"}

    timestamps = [log.timestamp.strftime("%Y-%m-%d") for log in logs]
    bmis = [log.bmi for log in logs]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, bmis, marker='o', color='blue')
    plt.title("Your BMI Over Time")
    plt.xlabel("Date")
    plt.ylabel("BMI")
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return {"bmi_chart_base64": encoded}
