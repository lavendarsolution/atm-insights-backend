from datetime import datetime, timedelta, timezone
from typing import Optional

from config import settings
from fastapi import HTTPException, status
from google.auth.transport import requests
from google.oauth2 import id_token
from jose import JWTError, jwt
from models.user import User, UserRole
from passlib.context import CryptContext
from schemas.auth import TokenData
from sqlalchemy.orm import Session

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    def __init__(self):
        self.secret_key = settings.jwt_secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60
        self.google_client_id = settings.google_client_id

    def verify_password(self, plain_password: str, password_hash: str) -> bool:
        return pwd_context.verify(plain_password, password_hash)

    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)

    def create_access_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.access_token_expire_minutes
            )

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> TokenData:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            email: str = payload.get("sub")
            if email is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                )
            return TokenData(email=email)
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
            )

    def authenticate_user(
        self, db: Session, email: str, password: str
    ) -> Optional[User]:
        user = db.query(User).filter(User.email == email).first()
        if not user or not user.password_hash:
            return None
        if not self.verify_password(password, user.password_hash):
            return None
        return user

    def verify_google_token(self, token: str) -> dict:
        try:
            idinfo = id_token.verify_oauth2_token(
                token, requests.Request(), self.google_client_id
            )
            return idinfo
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Google token"
            )

    def get_or_create_google_user(self, db: Session, google_data: dict) -> User:
        email = google_data.get("email")
        google_id = google_data.get("sub")
        full_name = google_data.get("name", "")

        # Check if user exists
        user = db.query(User).filter(User.email == email).first()

        if not user:
            # Create new user
            user = User(
                email=email,
                full_name=full_name,
                google_id=google_id,
                role=UserRole.USER,
                is_active=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        return user


auth_service = AuthService()
