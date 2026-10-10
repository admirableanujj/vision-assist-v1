# backend/app/core/security.py
import os
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "vision_assist_super_secret_jwt_key_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

def verify_password(plain_password: str, hashed_password_with_salt: str) -> bool:
    """
    Verifies a plain password against the stored 'salt:hash' string format.
    Matches the _hash_password helper in precheck_db.py.
    """
    try:
        salt, stored_hash = hashed_password_with_salt.split(":")
        salted_pass = plain_password.encode('utf-8') + salt.encode('utf-8')
        calculated_hash = hashlib.sha256(salted_pass).hexdigest()
        return calculated_hash == stored_hash
    except ValueError:
        # Invalid format in DB
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generates a signed JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
