import os
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
env_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../.env'))
load_dotenv(env_filename)

# --- Security Dependency ---
# This is a security dependency that looks for a Bearer token in the Authorization header
bearer_scheme = HTTPBearer()

# --- Models ---
class UserData(BaseModel):
    user_id: str
    email: Optional[str] = None

# --- JWT Validation Function ---
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> UserData:
    """
    Validates a JWT from the Authorization header and returns user data.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: Missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials

    # We use a try-except block to handle potential errors during JWT decoding
    try:
        # Fetch the public keys from Clerk's JWKS endpoint
        # In a production app, you would cache this to avoid repeated network calls.
        # The 'fastapi-clerk-middleware' library does this for you.
        import requests
        jwks = requests.get(CLERK_JWKS_URL).json()
        
        # Decode and verify the JWT. This will check the signature and expiration.
        payload = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],  # Clerk uses RS256 for its tokens
            audience="clerk",      # The audience claim should be 'clerk'
            issuer=CLERK_JWKS_URL.replace("/.well-known/jwks.json", "")
        )
        
        # Extract the user ID (the 'sub' claim) and other data
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: No user ID found",
            )
            
        return UserData(user_id=user_id, email=payload.get("email"))

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}",
        )

# --- FastAPI App ---
app = FastAPI()

# --- Protected Endpoint ---
@app.get("/users/me/")
async def read_users_me(user: UserData = Depends(get_current_user)):
    """
    A protected endpoint that can only be accessed with a valid Clerk session token.
    """
    # The 'user' object is available here, containing the user ID and other decoded info
    return {"message": "You are authenticated!", "user_id": user.user_id, "email": user.email}

# --- Public Endpoint ---
@app.get("/public")
async def public_endpoint():
    """
    A public endpoint that does not require any authentication.
    """
    return {"message": "This is a public endpoint."}

# --- Running the app ---
# uvicorn main:app --reload