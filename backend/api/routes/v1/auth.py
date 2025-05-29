from datetime import timedelta

from database import get_db
from dependencies.auth import get_current_active_user, require_admin
from fastapi import APIRouter, Depends, HTTPException, status
from models.user import User
from schemas.auth import (
    GoogleAuthRequest,
    LoginRequest,
    Token,
    UserCreate,
    UserResponse,
)
from services.auth_service import auth_service
from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/auth/login", response_model=Token)
async def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    user = auth_service.authenticate_user(db, login_data.email, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    access_token_expires = timedelta(minutes=auth_service.access_token_expire_minutes)
    access_token = auth_service.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@router.post("/auth/google", response_model=Token)
async def google_auth(google_data: GoogleAuthRequest, db: Session = Depends(get_db)):
    google_user_data = auth_service.verify_google_token(google_data.token)
    user = auth_service.get_or_create_google_user(db, google_user_data)

    access_token_expires = timedelta(minutes=auth_service.access_token_expire_minutes)
    access_token = auth_service.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@router.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Create new user
    password_hash = (
        auth_service.get_password_hash(user_data.password)
        if user_data.password
        else None
    )

    new_user = User(
        email=user_data.email,
        full_name=user_data.full_name,
        password_hash=password_hash,
        role=user_data.role,
        is_active=True,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return UserResponse.model_validate(new_user)


@router.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    return UserResponse.model_validate(current_user)


# Admin-only routes
@router.get("/auth/users", response_model=list[UserResponse])
async def get_all_users(
    admin_user: User = Depends(require_admin), db: Session = Depends(get_db)
):
    users = db.query(User).all()
    return [UserResponse.model_validate(user) for user in users]


@router.put("/auth/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    role: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.role = role
    db.commit()
    return {"message": "User role updated successfully"}
