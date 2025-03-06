from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

# Define mapping for renaming response keys
RESPONSE_KEY_MAPPING = {
    "question1": "favorite_cuisines",
    "question2": "allergies",    
    "question3": "food_preferences",
    "question4": "liked_ingredients",
    "question5": "sushi",
    "question6": "pizza",
    "question7": "dumpling",
    "question8": "hamburger",
    "question9": "fried_chicken",
    "question10": "taco",
    "question11": "pasta",
    "question12": "disliked_ingredients"
    
}

def remap_responses(responses: dict) -> dict:
    """Remap response keys based on the defined mapping."""
    remapped = {}
    for key, value in responses.items():
        new_key = RESPONSE_KEY_MAPPING.get(key, key)
        remapped[new_key] = value
    return remapped

async def save_survey_responses(survey_data: dict, db: AsyncIOMotorDatabase) -> str:
    """
    Process and save the survey responses into the database.
    Returns the inserted ID as a string.
    """
    # Remap responses keys
    survey_data["responses"] = remap_responses(survey_data.get("responses", {}))
    # Add a timestamp
    survey_data["submitted_at"] = datetime.utcnow()
    
    # Insert into the "surveys" collection
    result = await db.surveys.insert_one(survey_data)
    return str(result.inserted_id)
