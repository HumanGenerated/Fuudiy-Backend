from fastapi import APIRouter, Depends, Request
from motor.motor_asyncio import AsyncIOMotorDatabase
from server.services.auth_service import register_user, login_user
from server.models.auth import UserCreate, UserLogin, Token

router = APIRouter()

# Dependency to get the database from the app state
async def get_db(request: Request) -> AsyncIOMotorDatabase:
    return request.app.state.database

@router.post("/register", response_model=Token, tags=["Auth"])
async def register(user: UserCreate, db: AsyncIOMotorDatabase = Depends(get_db)):
    return await register_user(user, db)

@router.post("/login", response_model=Token, tags=["Auth"])
async def login(user: UserLogin, db: AsyncIOMotorDatabase = Depends(get_db)):
    return await login_user(user, db)
