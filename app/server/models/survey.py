from pydantic import BaseModel
from typing import Any, Dict, Optional

class SurveyResponse(BaseModel):
    user_id: Optional[str] = None  # Optionally tie survey responses to a user (e.g., via JWT)
    responses: Dict[str, Any]
