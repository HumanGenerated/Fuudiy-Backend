from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------ Routers -----------------------------
from server.routes.food import router as FoodRouter
from server.routes.user import router as UserRouter
from server.routes.comment import router as UserCommentsRouter
from server.routes.auth import router as AuthRouter
from server.routes.survey import router as SurveyRouter
from server.routes.translation import router as TranslationRouter
from server.routes.connection import router as ConnectionRouter
from server.routes.explore import router as ExploreRouter
from server.routes.search import router as SearchRouter   # ⬅️
from server.database import database
from spark_utils import spark
# --------------------------------------------------------------------

app = FastAPI()            # ← keep it simple; /api is added by root-path parameter in uvicorn

# --------------------------------------------------------------------
# 1.  CORS: we only need this when the browser is *NOT* on the same
#     origin.  In production the proxy gives one origin, so allow
#     localhost for dev and fuudiy.com for good measure.
# --------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # CRA dev server
        "http://127.0.0.1:3000",
        "https://fuudiy.com",      # optional – same origin means CORS is unused
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Authorization"],
)

# --------------------------------------------------------------------
# 2.  Include routers – **no /api prefix here** because the proxy (or
#     uvicorn --root-path /api) puts /api in front of everything.
#     The one router that *already* used "/api" now becomes "/search".
# --------------------------------------------------------------------
app.include_router(FoodRouter,       prefix="/food",        tags=["Food"])
app.include_router(UserRouter,       prefix="/users",       tags=["User"])
app.include_router(UserCommentsRouter, prefix="/comments",  tags=["Comment"])
app.include_router(AuthRouter,       prefix="/auth",        tags=["Auth"])
app.include_router(SurveyRouter,     prefix="/survey",      tags=["Survey"])
app.include_router(TranslationRouter,prefix="/translation", tags=["Translation"])
app.include_router(ExploreRouter,    prefix="/explore",     tags=["Explore"])
app.include_router(ConnectionRouter, prefix="/connections", tags=["Connection"])

# 🔄 CHANGED: was "/api"; avoid /api/api duplication
app.include_router(SearchRouter,     prefix="/search",      tags=["SearchBar"])

# --------------------------------------------------------------------
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to this fantastic app!"}

# give other modules easy access to the database handle
app.state.database = database

@app.on_event("shutdown")
def shutdown_event():
    spark.stop()
    print("Spark session stopped")
