# backend/app/api/auth_routes.py
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import APIRouter, HTTPException, status, Depends
from app.models.schemas import LoginRequest, TokenResponse, UserResponse
from app.core.security import verify_password, create_access_token
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

def get_db_connection():
    """Dependency or helper to acquire a PostgreSQL connection."""
    db_host = os.getenv("POSTGRES_HOST", "vision_assist_db")
    db_name = os.getenv("POSTGRES_DB", "vision_assist")
    db_user = os.getenv("POSTGRES_USER", "postgres")
    
    secret_path = "/run/secrets/postgres_password"
    if os.path.exists(secret_path):
        with open(secret_path, "r") as f:
            db_password = f.read().strip()
    else:
        db_password = os.getenv("POSTGRES_PASSWORD", "postgres")

    try:
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to DB in auth_routes: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service currently unavailable."
        )

@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest):
    """
    Authenticates user credentials, logs login attempt history,
    and returns a signed JWT bearer token.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 1. Fetch user and hashed password from users + user_login tables
            query = """
                SELECT 
                    u.id, u.guid_user, u.username, u.first_name, 
                    u.last_name, u.email, u.role, u.is_active,
                    ul.hashed_password
                FROM users u
                JOIN user_login ul ON u.id = ul.user_id
                WHERE u.username = %s AND u.is_active = TRUE;
            """
            cur.execute(query, (credentials.username,))
            user_data = cur.fetchone()

            # 2. Handle invalid username or inactive account
            if not user_data:
                logger.warning(f"Failed login attempt: Username '{credentials.username}' not found.")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="The Genie rejects these credentials."
                )

            # 3. Verify password hash match
            is_valid = verify_password(credentials.password, user_data["hashed_password"])

            # 4. Log login history
            login_status = "SUCCESS" if is_valid else "FAILED_PASSWORD"
            cur.execute("""
                INSERT INTO user_login_history (user_id, login_status, last_modified_by)
                VALUES (%s, %s, %s);
            """, (user_data["id"], login_status, credentials.username))
            conn.commit()

            if not is_valid:
                logger.warning(f"Failed login attempt for user: {credentials.username}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="The Genie rejects these credentials."
                )

            # 5. Create JWT Token payload
            user_response = UserResponse(
                id=user_data["id"],
                guid_user=user_data["guid_user"],
                username=user_data["username"],
                first_name=user_data["first_name"],
                last_name=user_data["last_name"],
                email=user_data["email"],
                role=user_data["role"],
                is_active=user_data["is_active"]
            )

            access_token = create_access_token(
                data={"sub": user_data["username"], "user_id": user_data["id"], "role": user_data["role"]}
            )

            logger.info(f"User successfully authenticated: {credentials.username}")
            return TokenResponse(access_token=access_token, user=user_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during user authentication: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while granting passage."
        )
    finally:
        conn.close()