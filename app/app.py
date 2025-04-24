from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.routes.food import router as FoodRouter
from server.routes.user import router as UserRouter
from server.routes.comment import router as UserCommentsRouter
from server.routes.auth import router as AuthRouter
from server.routes.survey import router as SurveyRouter
from server.routes.translation import router as TranslationRouter
from server.routes.connection import router as ConnectionRouter
from server.database import database
from server.routes.explore import router as ExploreRouter
from server.routes.search import router as SearchRouter
from spark_utils import spark
#from server.routes.similarity_explore import

#from server.routes.translation import router as TranslationRouter
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"
                   ,"https://accounts.google.com" ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Authorization"],
)

# Include routers with unique prefixes
app.include_router(FoodRouter, prefix="/food", tags=["Food"])
app.include_router(UserRouter, prefix="/users", tags=["User"])
app.include_router(UserCommentsRouter, prefix="/comments", tags=["Comment"])
app.include_router(AuthRouter, prefix="/auth", tags=["Auth"])
app.include_router(SurveyRouter, prefix="/survey", tags=["Survey"])
app.include_router(TranslationRouter, prefix="/translation", tags=["Translation"])
app.include_router(ExploreRouter, prefix="/explore", tags=["Explore"])
app.include_router(ConnectionRouter, prefix="/connections", tags=["Connection"])
app.include_router(SearchRouter, prefix="/api", tags=["SearchBar"])

#app.include_router(TranslationRouter, prefix="/translation", tags=["Translation"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to this fantastic app!"}

app.state.database = database
@app.on_event("shutdown")
def shutdown_event():
    spark.stop()  # Properly stop Spark session
    print("Spark session stopped")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    