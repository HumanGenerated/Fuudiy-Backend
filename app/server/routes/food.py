import os
from pathlib import Path  
from google.cloud import storage
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from bson import ObjectId
from server.database import database
from server.services.food_service import (
    add_food,
    delete_food,
    retrieve_food,
    retrieve_foods,
    retrieve_first_10_foods,
    update_food,
    get_top_5_food,
    get_top_rated_foods_by_cuisine,  
)
from server.models.food import (
    ErrorResponseModel,
    ResponseModel,
    FoodSchema,
    UpdateFoodModel,
)
from server.services.comment_service import update_rate_for_comment  # ✅ FIXED: Import this function


current_file = Path(__file__)
credentials_path = current_file.parents[3] / "gcs-key.json"

router = APIRouter()
food_collection = database.get_collection("foods")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
# Google Cloud Storage details
BUCKET_NAME = "fuudiy_bucket"

@router.get("/top-5-foods", tags=["Food"], response_model=list)
async def fetch_top_5_foods():
    try:
        foods = await get_top_5_food()
        if not foods:
            raise HTTPException(status_code=404, detail="No food items found.")
        return foods
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/top-foods-by-country", tags=["Food"], response_model=list)
async def fetch_top_foods_by_countries():
    try:
        foods = await get_top_rated_foods_by_cuisine()
        if not foods:
            raise HTTPException(status_code=404, detail="No food items found.")
        return foods
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", tags=["Food"], response_description="Food data added into the database")
async def add_food_data(food: FoodSchema = Body(...)):
    food = jsonable_encoder(food)
    new_food = await add_food(food)
    return ResponseModel(new_food, "Food added successfully.")


# @router.get("/", response_description="Foods retrieved")
# async def get_foods():
#     foods = await retrieve_first_10_foods()
#     if foods:
#         return ResponseModel(foods, "Food data retrieved successfully")
#     return ResponseModel(foods, "Empty list returned")


@router.get("/{id}", tags=["Food"])
async def get_food(id: str):
    try:
        food = await retrieve_food(id)
        if not food:
            raise HTTPException(status_code=404, detail="Food not found")
        return jsonable_encoder(food)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Get image url

def get_image_url(image_id):
    """Generate a signed URL to access a private image in GCS."""
    try:
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set")

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{image_id}.png")
        
        if not blob.exists():
            print(f"Image {image_id}.png not found in bucket {BUCKET_NAME}")
            return None  

        # Generate a signed URL valid for 1 hour
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=3600,  # 1 hour expiration
            method="GET"
        )
        return signed_url

    except Exception as e:
        print(f"Error generating signed URL: {str(e)}")
        return None

@router.get("/image/{image_id}")
async def fetch_image(image_id: str):
    url = get_image_url(image_id)
    if not url:
        raise HTTPException(status_code=404, detail="Image not found in GCS")

    return {"image_url": url}

@router.put("/update-rating/{user_id}/{food_id}", tags=["Food"], response_description="Update the food rating")
async def update_food_rating(user_id: str, food_id: str, new_rate: int = Query(..., ge=1, le=5)):
    """
    Updates the user's rating for a food item and recalculates the food's popularity.
    The rate must be between 1 and 5.
    """

    try:
        # ✅ Convert IDs to ObjectId
        user_obj_id = ObjectId(user_id)
        food_obj_id = ObjectId(food_id)

        # ✅ Find the user's comment
        user_comment = await database.get_collection("user_comments").find_one(
            {"userId": user_obj_id, "foodId": food_obj_id}
        )

        if not user_comment:
            raise HTTPException(status_code=404, detail="User has not rated this food yet")

        old_rate = user_comment.get("rate", 0)

        # ✅ Find the food item
        food = await database.get_collection("foods").find_one({"_id": food_obj_id})
        if not food:
            raise HTTPException(status_code=404, detail="Food not found")

        # ✅ Get current popularity details
        popularity = food.get("popularity", {"rating": 0, "votes": 0})
        current_rating = popularity.get("rating", 0)
        votes = popularity.get("votes", 0)

        # ✅ Recalculate the new rating
        if votes > 0:
            new_rating = ((current_rating * votes) - old_rate + new_rate) / votes
        else:
            new_rating = new_rate  # If no previous votes, take new rate as rating

        # ✅ Update the food's popularity
        await database.get_collection("foods").update_one(
            {"_id": food_obj_id},
            {"$set": {"popularity.rating": new_rating}}
        )

        # ✅ Update the user's rating in the `user_comments` collection
        update_successful = await update_rate_for_comment(user_id, food_id, new_rate)

        if not update_successful:
            raise HTTPException(status_code=500, detail="Failed to update the user's rating.")

        return {"message": f"Rating updated successfully. New food rating: {new_rating:.2f}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/update-popularity/{food_id}", tags=["Food"], response_description="Update food popularity")
async def update_food_popularity(food_id: str, data: dict = Body(...)):
    """
    Updates the popularity rating of a food item.
    Accepts only `rating` in the body.
    Automatically updates votes and recalculates the new rating.
    If `existing_vote` is True, votes remain the same. Otherwise, votes increase by 1.
    """
    try:
        print(f"Received request: food_id={food_id}, data={data}")

        # Validate food_id
        if not ObjectId.is_valid(food_id):
            raise HTTPException(status_code=400, detail="Invalid food_id format")

        food_obj_id = ObjectId(food_id)

        # Extract rating and existing_vote from the body
        new_rating = data.get("rating")
        existing_vote = data.get("existing_vote", False)  # Default to False

        if new_rating is None:
            raise HTTPException(status_code=400, detail="Missing required field: rating.")

        print(f"Parsed values: new_rating={new_rating}, existing_vote={existing_vote}")

        # Retrieve the current food document
        food = await food_collection.find_one({"_id": food_obj_id})
        if not food:
            raise HTTPException(status_code=404, detail="Food not found.")

        print(f"Existing food data: {food}")

        # Get current votes
        current_votes = food.get("popularity", {}).get("votes", 0)

        # Determine new vote count
        new_votes = current_votes if existing_vote else current_votes + 1

        # Update the database
        update_data = {
            "popularity.rating": round(new_rating, 1),
            "popularity.votes": new_votes
        }

        print(f"Updating database with: {update_data}")

        result = await food_collection.update_one({"_id": food_obj_id}, {"$set": update_data})

        if result.modified_count == 0:
            raise HTTPException(status_code=500, detail="Failed to update food popularity.")

        return {
            "message": f"Food {food_id} popularity updated successfully.",
            "new_rating": round(new_rating, 1),
            "votes": new_votes
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
