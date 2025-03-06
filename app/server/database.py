from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGO_URI, DATABASE_NAME
from fastapi import FastAPI

if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in the .env file")

client = AsyncIOMotorClient(MONGO_URI)

database = client[DATABASE_NAME]