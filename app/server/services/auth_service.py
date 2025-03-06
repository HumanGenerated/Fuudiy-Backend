from motor.motor_asyncio import AsyncIOMotorDatabase
from fastapi import HTTPException, status, Depends
from server.models.auth import UserCreate, UserLogin
from server.models.auth import get_password_hash, verify_password, create_access_token, TokenData
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from config import SECRET_KEY

ALGORITHM = "HS256"

print("SECRET_KEY:", SECRET_KEY)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def verify_access_token(token: str):
    print("token",token)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print("payload",payload)
        token_data = TokenData(**{"user_id": payload.get("user_id")})
        if token_data.user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return token_data.user_id
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
def get_current_user(token: str = Depends(oauth2_scheme)):
    return verify_access_token(token)


async def register_user(user: UserCreate, db: AsyncIOMotorDatabase):
    existing_user = await db.users.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user_data = user.dict()
    user_data["password"] = get_password_hash(user.password)
    result = await db.users.insert_one(user_data)
    user_id = str(result.inserted_id)
    access_token = create_access_token(data={"user_id": user_id})
    print("id: ",user_id)
    print("access_token ",access_token)
    return {"access_token": access_token, "token_type": "bearer"}

async def login_user(user: UserLogin, db: AsyncIOMotorDatabase):
    existing_user = await db.users.find_one({"email": user.email})
    if not existing_user or not verify_password(user.password, existing_user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    user_id = str(existing_user["_id"])
    access_token = create_access_token(data={"user_id": user_id})

    return {"access_token": access_token, "token_type": "bearer"}
