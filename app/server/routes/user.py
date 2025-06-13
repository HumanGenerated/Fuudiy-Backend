from fastapi import APIRouter, Body, HTTPException, Depends
from fastapi.encoders import jsonable_encoder
from server.database import database
from ..services.auth_service import get_current_user
from server.services.user_service import (
    add_user,
    delete_user,
    retrieve_user,
    retrieve_current_user,
    retrieve_users,
    update_user,
)
from server.models.user import (
    DislikedIngredientsUpdateModel,
    AllergiesUpdateModel,
    ErrorResponseModel,
    ResponseModel,
    UserSchema,
    UpdateUserModel,
)

router = APIRouter() 
user_collection = database.get_collection("users")
survey_collection = database.get_collection("surveys")

@router.get("/", tags=["User"], response_description="Get all users")
async def get_all_users():
    """
    Fetch all users from the database.
    """
    try:
        users = await retrieve_users()
        if not users:
            return ResponseModel([], "No users found in the database.")
        return ResponseModel(users, "Users retrieved successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{username}", tags=["User"], response_description="Get a specific user by ID")
async def get_user(username: str):
    """
    Fetch a specific user by their ID.
    """
    #user_id: str = Depends(get_current_user)
    user = await retrieve_user(username)
    if user:
        return ResponseModel(user, f"User with ID {id} retrieved successfully.")
    raise HTTPException(status_code=404, detail=f"User with ID {id} not found")

@router.post("/me", tags=["User"], response_description="Get authenticated user's info and preferences")
async def get_current_user_info(user_id: str = Depends(get_current_user)):
    try:
        user = await retrieve_user(user_id)
        survey = await survey_collection.find_one({"user_id": user_id})

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        responses = survey.get("responses", {}) if survey else {}

        return ResponseModel(
            {
                **user,
                **responses  # Merges dislikedIngredients, etc.
            },
            f"User with ID {user_id} and preferences retrieved successfully."
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
    
@router.post("/", tags=["User"], response_description="Add a new user to the database")
async def add_user_data(user: UserSchema = Body(...)):
    """
    Add a new user to the database.
    """
    user = jsonable_encoder(user)
    new_user = await add_user(user)
    return ResponseModel(new_user, "User added successfully.")

@router.put("/{id}", tags=["User"], response_description="Update user data by ID")
async def update_user_data(id: str, req: UpdateUserModel = Body(...)):
    """
    Update a user's data by their ID.
    """
    req = {k: v for k, v in req.dict().items() if v is not None}
    updated_user = await update_user(id, req)
    if updated_user:
        return ResponseModel(f"User with ID {id} updated successfully.", "Success")
    raise HTTPException(status_code=404, detail=f"User with ID {id} not found")


@router.put("/update-avatar/{id}", tags=["User"], response_description="Update user profile picture by ID")
async def update_user_avatar(id: str, req: dict = Body(...)):
    """
    Update the user's profile picture.
    Expects a JSON payload like: {"avatarId": "newAvatarName"}
    """
    if "avatarId" not in req:
        raise HTTPException(status_code=400, detail="avatarId is required")
    updated = await update_user(id, {"avatarId": req["avatarId"]})
    if updated:
        return ResponseModel(f"User with ID {id} avatar updated successfully.", "Success")
    raise HTTPException(status_code=404, detail=f"User with ID {id} not found")

@router.put("/update-disliked", tags=["User"], response_description="Update disliked ingredients using token")
async def update_user_disliked(
    req: DislikedIngredientsUpdateModel,
    user_id: str = Depends(get_current_user)
):
    """
    Update the user's disliked ingredients using the token.
    """
    disliked_str = ", ".join(req.dislikedIngredients)

    updated = await update_user(user_id, {"disliked_ingredients": disliked_str})
    if updated:
        return ResponseModel(f"User with ID {user_id} disliked ingredients updated successfully.", "Success")
    
    raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")

@router.put("/update-disliked-by-username/{username}", tags=["User"], response_description="Update disliked ingredients by username")
async def update_disliked_by_username(
    username: str,
    req: DislikedIngredientsUpdateModel = Body(...)
):
    """
    Update the user's disliked ingredients in the 'surveys' collection using their username.
    Expects: {"dislikedIngredients": ["Onion", "Tomato"]}
    """
    disliked_str = ", ".join(req.dislikedIngredients)

    # Get user
    user = await user_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail=f"User with username '{username}' not found")

    user_id = str(user["_id"])

    # Update disliked_ingredients in surveys collection
    result = await survey_collection.update_one(
        {"user_id": user_id},
        {"$set": {"responses.disliked_ingredients": disliked_str}}
    )

    if result.modified_count == 0:
        return ResponseModel(f"No change detected for user '{username}'", "No Update")

    return ResponseModel(f"User '{username}' disliked ingredients updated in surveys.", "Success")


@router.delete("/{id}", tags=["User"], response_description="Delete a user by ID")
async def delete_user_data(id: str):
    """
    Delete a user by their ID.
    """
    deleted_user = await delete_user(id)
    if deleted_user:
        return ResponseModel(f"User with ID {id} deleted successfully.", "Success")
    raise HTTPException(status_code=404, detail=f"User with ID {id} not found")

@router.put("/update-avatar-by-username/{username}", tags=["User"], response_description="Update user avatar by username")
async def update_avatar_by_username(username: str, req: dict = Body(...)):
    """
    Update the user's avatar using their username.
    Expects: {"avatarId": "avatarName"}
    """
    if "avatarId" not in req:
        raise HTTPException(status_code=400, detail="avatarId is required")

    user = await user_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail=f"User with username '{username}' not found")

    updated = await user_collection.update_one(
        {"username": username},
        {"$set": {"avatarId": req["avatarId"]}}
    )
    
    if updated.modified_count > 0:
        return ResponseModel(f"Avatar for user '{username}' updated successfully.", "Success")
    
    return ResponseModel(f"No change detected for user '{username}'.", "No Update")

@router.get("/preferences/{user_id}", tags=["User"], response_description="Get user food preferences")
async def get_user_preferences(user_id: str):
    """
    Fetch food preferences from the survey for a given user.
    """
    survey = await survey_collection.find_one({"user_id": user_id})
    print("routes user get_user_preferences survey: ", survey)
    if not survey or "responses" not in survey:
        raise HTTPException(status_code=404, detail="Survey data not found for user")
    return ResponseModel(survey["responses"], "Survey preferences retrieved.")

@router.put("/update-bio-by-username/{username}", tags=["User"], response_description="Update user bio by username")
async def update_bio_by_username(username: str, req: dict = Body(...)):
    """
    Update the user's bio using their username.
    Expects: {"bio": "This is my new bio"}
    """
    if "bio" not in req:
        raise HTTPException(status_code=400, detail="bio is required")

    user = await user_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail=f"User with username '{username}' not found")

    updated = await user_collection.update_one(
        {"username": username},
        {"$set": {"bio": req["bio"]}}
    )

    if updated.modified_count > 0:
        return ResponseModel(f"Bio for user '{username}' updated successfully.", "Success")

    return ResponseModel(f"No change detected for user '{username}'.", "No Update")

@router.get("/allergies/{username}", tags=["User"], response_description="Get user allergies by username")
async def get_user_allergies_by_username(username: str):
    """
    Fetch allergy information from the survey collection for a user by username.
    """
    # Find user by username
    user = await user_collection.find_one({"username": username})
    print("user/allergies/{username} get_user_allergies_by_username user: ", user)
    if not user:
        raise HTTPException(status_code=404, detail=f"User with username '{username}' not found")

    user_id = str(user["_id"])

    # Find survey responses for this user
    survey = await survey_collection.find_one({"user_id": user_id})
    print("user/allergies/{username} get_user_allergies_by_username survey: ", survey)
    if not survey or "responses" not in survey:
        return ResponseModel([], f"No survey data found for user '{username}', returning empty allergy list.")


    allergies = survey["responses"].get("allergies", [])
    return ResponseModel(allergies, f"Allergies for user '{username}' retrieved successfully.")

@router.put("/update-allergies-by-username/{username}", tags=["User"], response_description="Update allergies by username")
async def update_allergies_by_username(username: str, req: AllergiesUpdateModel = Body(...)):
    """
    Update the user's allergy list in the 'surveys' collection using their username.
    Expects: {"allergies": ["eggs", "dairy"]}
    """
    user = await user_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail=f"User with username '{username}' not found")
    
    user_id = str(user["_id"])

    result = await survey_collection.update_one(
        {"user_id": user_id},
        {"$set": {"responses.allergies": req.allergies}}
    )

    if result.modified_count == 0:
        return ResponseModel(f"No change detected for allergies of user '{username}'", "No Update")

    return ResponseModel(f"Allergies for user '{username}' updated in surveys.", "Success")


@router.put("/update-disliked-by-username/{username}", tags=["User"], response_description="Update disliked ingredients by username")
async def update_disliked_by_username(username: str, req: DislikedIngredientsUpdateModel = Body(...)):
    """
    Update the user's disliked ingredients in the 'surveys' collection using their username.
    Expects: {"dislikedIngredients": ["Onion", "Tomato"]}
    """
    disliked_str = ", ".join(req.dislikedIngredients)

    # Get user
    user = await user_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail=f"User with username '{username}' not found")

    user_id = str(user["_id"])

    # Update disliked_ingredients in surveys collection
    result = await survey_collection.update_one(
        {"user_id": user_id},
        {"$set": {"responses.disliked_ingredients": disliked_str}}
    )

    if result.modified_count == 0:
        return ResponseModel(f"No change detected for user '{username}'", "No Update")

    return ResponseModel(f"User '{username}' disliked ingredients updated in surveys.", "Success")
