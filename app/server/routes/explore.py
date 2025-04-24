import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from ..services.auth_service import get_current_user
import traceback
from pymongo import MongoClient
from bson import ObjectId
from config import MONGO_URI, DATABASE_NAME, COLLECTION_FOOD, COLLECTION_SURVEY, COLLECTION_USER_COMMENTS

router = APIRouter()

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

DISH_INGREDIENT_MAP = {
    "vegetarian_dishes": ["lettuce", "tomato"],
    "meat_dishes": ["meat", "lamb", "beef", "steak"],
    "seafood": ["lobster", "crab", "fish", "shellfish"],
    "pastries_and_bread": ["flour", "yeast"],
    "sweets_and_confectionery": ["sugar", "chocolate"],
    "sushi": ["rice", "fish", "nori", "tuna", "crab", "soy sauce"],
    "pizza": ["cheese", "tomato", "flour", "saussage"],
    "dumpling": ["flour", "meat", "vegetables"],
    "hamburger": ["meat", "lettuce", "tomato", "bread"],
    "fried_chicken": ["chicken", "flour", "spices"],
    "taco": ["tortilla", "meat", "lettuce", "tomato"],
    "pasta": ["pasta", "tomato", "cheese"]
}

RATING_ADJUSTMENTS = {
    5: 2.0,
    4: 1.0,
    3: 0.0,
    2: -1.0,
    1: -2.0
}

def convert_objectid_to_str(obj: Any) -> Any:
    """Convert ObjectId to string in nested dictionaries"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_objectid_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid_to_str(item) for item in obj]
    return obj

def load_collection(coll_name: str) -> List[Dict]:
    """Load data from MongoDB collection"""
    return list(db[coll_name].find())

def parse_prefs(value: Any) -> List[str]:
    """Parse preferences from string or list"""
    if isinstance(value, str):
        return [x.strip().lower() for x in value.split(",") if x.strip()]
    return value if isinstance(value, list) else []

def filter_disliked_allergies(foods: List[Dict], disliked: List[str], allergies: Optional[List[str]] = None) -> List[Dict]:
    """Filter foods based on disliked ingredients and allergies"""
    all_exclusions = set(disliked)
    if allergies:
        all_exclusions.update(allergies)
    
    if not all_exclusions:
        return foods

    return [
        f for f in foods
        if f.get('ingredients')
        and not any(ing in all_exclusions for ing in f.get('ingredients', []))
    ]

def calculate_ingredient_adjustments(prefs: Dict) -> Dict[str, float]:
    """Calculate ingredient adjustments based on user preferences"""
    adjustments = defaultdict(float)
    
    food_prefs = prefs.get("food_preferences", {})
    
    if isinstance(food_prefs, list):
        food_prefs = {item["key"]: item["value"] for item in food_prefs}
    
    for dish, ingredients in DISH_INGREDIENT_MAP.items():
        raw_rating = prefs.get(dish) or food_prefs.get(dish)
        
        numerical_rating = 3
        
        if raw_rating:
            if isinstance(raw_rating, (int, float)):
                numerical_rating = max(1, min(int(raw_rating), 5))
            elif isinstance(raw_rating, str):
                clean_rating = raw_rating.strip().lower()
                numerical_rating = {
                    "loves": 5, 
                    "likes": 3, 
                    "neutral": 3,
                    "dislikes": 1
                }.get(clean_rating, 3)

        adj = RATING_ADJUSTMENTS.get(numerical_rating, 0.0)
        for ingredient in ingredients:
            adjustments[ingredient] += adj
            
    return adjustments

def calculate_tfidf_scores(foods: List[Dict]) -> List[Dict]:
    """Calculate TF-IDF scores for foods based on ingredients"""
    # Create a vocabulary of all ingredients
    all_ingredients = set()
    for food in foods:
        all_ingredients.update(food.get('ingredients', []))
    vocab = list(all_ingredients)
    
    # Calculate document frequency
    df = defaultdict(int)
    for food in foods:
        for ing in set(food.get('ingredients', [])):
            df[ing] += 1
    
    # Calculate TF-IDF scores
    for food in foods:
        ingredients = food.get('ingredients', [])
        tfidf_score = 0.0
        for ing in ingredients:
            tf = ingredients.count(ing) / len(ingredients)
            idf = np.log(len(foods) / (df[ing] + 1))
            tfidf_score += tf * idf
        food['tfidf_score'] = tfidf_score
    
    return foods

@router.get("/recommend/")
async def recommend_foods(
    country: str = Query(..., title="Target country"),
    diet: Optional[str] = Query(None, title="Dietary restriction"),
    user_id: str = Depends(get_current_user),
    top_n: int = 10
):
    try:
        # Load data
        all_foods = load_collection(COLLECTION_FOOD)
        all_surveys = load_collection(COLLECTION_SURVEY)
        all_comments = load_collection(COLLECTION_USER_COMMENTS)

        # Get user preferences
        user_survey = next((s for s in all_surveys if s.get('user_id') == user_id), None)
        if not user_survey:
            raise HTTPException(400, "Survey missing")
        prefs = user_survey.get("responses", {})

        # Prepare allergy/diet filters
        exclusion_ingredients = set()
        if diet:
            restrictions = db["special_diet"].find_one({"category": diet.lower()})
            exclusion_ingredients = set(restrictions.get("restricted_ingredients", [])) if restrictions else set()

        disliked = set(parse_prefs(prefs.get("disliked_ingredients", [])))
        allergies = set(parse_prefs(prefs.get("allergies", [])))
        all_exclusions = disliked | allergies | exclusion_ingredients

        # Calculate ingredient adjustments
        ingredient_adjustments = calculate_ingredient_adjustments(prefs)

        # Filter and score foods
        filtered_foods = [
            f for f in all_foods
            if f.get('country') == country
            and f.get('ingredients')
            and not any(ing in all_exclusions for ing in f.get('ingredients', []))
        ]

        # Calculate TF-IDF scores
        filtered_foods = calculate_tfidf_scores(filtered_foods)

        # Add user rating adjustments
        user_comments = [c for c in all_comments if c.get('userId') == user_id]
        for food in filtered_foods:
            food_comments = [c for c in user_comments if str(c.get('foodId')) == str(food['_id'])]
            rating_adjustment = sum((float(c.get('rate', 3)) - 3) * 1.0 for c in food_comments)
            ingredient_adjustment = sum(ingredient_adjustments.get(ing, 0.0) for ing in food.get('ingredients', []))
            food['score'] = food.get('tfidf_score', 0.0) + ingredient_adjustment + rating_adjustment

        # Get similar user recommendations
        similar_users = []
        user_high_rated_foods = {str(c.get('foodId')) for c in user_comments if float(c.get('rate', 0)) >= 4}
        
        for comment in all_comments:
            if comment.get('userId') != user_id and float(comment.get('rate', 0)) >= 4:
                if str(comment.get('foodId')) in user_high_rated_foods:
                    similar_users.append(comment.get('userId'))

        similar_users_ratings = [
            c for c in all_comments
            if c.get('userId') in similar_users
            and float(c.get('rate', 0)) >= 4
            and str(c.get('foodId')) not in user_high_rated_foods
        ]

        similar_foods = []
        for food_id in {c.get('foodId') for c in similar_users_ratings}:
            food = next((f for f in all_foods if str(f['_id']) == str(food_id)), None)
            if food and food.get('country') == country:
                similar_foods.append(food)

        # Sort and return results
        filtered_foods.sort(key=lambda f: f['score'], reverse=True)
        similar_foods.sort(key=lambda f: f.get('score', 0), reverse=True)

        # Convert ObjectId to string before returning
        filtered_foods = convert_objectid_to_str(filtered_foods[:top_n])
        similar_foods = convert_objectid_to_str(similar_foods[:5])

        return {
            "personalized_recommendations": filtered_foods,
            "similar_users_recommendations": similar_foods
        }

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Recommendation engine failed: {e}")

def cosine_similarity(list1: List[str], list2: List[str]) -> float:
    """Calculate cosine similarity between two lists of ingredients"""
    all_ings = list(set(list1) | set(list2))
    v1 = np.array([1 if ing in list1 else 0 for ing in all_ings])
    v2 = np.array([1 if ing in list2 else 0 for ing in all_ings])
    num = np.dot(v1, v2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(num) / denom if denom != 0 else 0.0

@router.get("/similar/{food_id}")
async def get_similar_foods(
    food_id: str,
    country: str = Query(..., title="Target country"),
    diet: Optional[str] = Query(None, title="Dietary restriction"),
    user_id: str = Depends(get_current_user),
    top_n: int = 10
):
    try:
        all_foods = load_collection(COLLECTION_FOOD)
        target_food = next((f for f in all_foods if str(f['_id']) == food_id), None)
        if not target_food:
            raise HTTPException(404, "Food not found")
        target_ings = target_food.get("ingredients", [])

        # Get user preferences
        all_surveys = load_collection(COLLECTION_SURVEY)
        user_survey = next((s for s in all_surveys if s.get('user_id') == user_id), None)
        if not user_survey:
            raise HTTPException(400, "Survey missing")
        prefs = user_survey.get("responses", {})

        # Prepare exclusions
        exclusion_ingredients = set()
        if diet:
            restrictions = db["special_diet"].find_one({"category": diet.lower()})
            exclusion_ingredients = set(restrictions.get("restricted_ingredients", [])) if restrictions else set()
        allergies = set(parse_prefs(prefs.get("allergies", [])))
        all_exclusions = allergies | exclusion_ingredients

        # Filter and score similar foods
        candidate_foods = [
            f for f in all_foods
            if f.get('country') == country
            and str(f['_id']) != food_id
            and f.get('ingredients')
            and not any(ing in all_exclusions for ing in f.get('ingredients', []))
        ]

        for f in candidate_foods:
            f['similarity'] = cosine_similarity(target_ings, f.get('ingredients', []))

        candidate_foods.sort(key=lambda f: f['similarity'], reverse=True)

        # Convert ObjectId to string before returning
        candidate_foods = convert_objectid_to_str(candidate_foods[:top_n])
        
        return {
            "results": candidate_foods
        }

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(500, str(e))