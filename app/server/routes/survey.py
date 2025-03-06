from fastapi import APIRouter, HTTPException, status, Depends, Request
from motor.motor_asyncio import AsyncIOMotorDatabase
from ..models.survey import SurveyResponse
from ..services.survey_service import save_survey_responses
from ..services.auth_service import get_current_user  # Import your token verification dependency

router = APIRouter()

async def get_db(request: Request) -> AsyncIOMotorDatabase:
    return request.app.state.database

@router.post("/submit", status_code=status.HTTP_201_CREATED)
async def submit_survey(
    survey: SurveyResponse,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user_id: str = Depends(get_current_user)  # Protect the endpoint
):
    survey_data = survey.dict()
    # Optionally, you can associate the survey with the user submitting it
    survey_data["user_id"] = user_id

    inserted_id = await save_survey_responses(survey_data, db)
    
    if inserted_id:
        return {
            "message": "Survey responses saved",
            "survey_id": inserted_id
        }
    
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to save survey responses"
    )
