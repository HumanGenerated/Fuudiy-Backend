import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List, Dict, Any
from collections import defaultdict
import traceback
from pymongo import MongoClient
from bson import ObjectId

# Configuration imports
from config import MONGO_URI, DATABASE_NAME, COLLECTION_FOOD, COLLECTION_SURVEY, COLLECTION_USER_COMMENTS
from ..services.auth_service import get_current_user

router = APIRouter()

# Database connection
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# Constants
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

# Cache for dataframes to avoid repeated database queries
_cache = {}

def load_food_data(collection_name):
    """Load data from MongoDB into pandas DataFrame"""
    if collection_name in _cache:
        return _cache[collection_name]
    
    collection = db[collection_name]
    data = list(collection.find())
    
    # Convert ObjectId to string for JSON serialization
    for item in data:
        if '_id' in item and isinstance(item['_id'], ObjectId):
            item['_id'] = str(item['_id'])
        
        # Also convert any other ObjectId fields that might exist
        for key, value in item.items():
            if isinstance(value, ObjectId):
                item[key] = str(value)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, ObjectId):
                        value[k] = str(v)
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, ObjectId):
                        value[i] = str(v)
                    elif isinstance(v, dict):
                        for k2, v2 in v.items():
                            if isinstance(v2, ObjectId):
                                v[k2] = str(v2)
    
    df = pd.DataFrame(data)
    _cache[collection_name] = df
    return df

def parse_prefs(value):
    """Parse comma-separated preferences into a list"""
    if isinstance(value, str):
        return [x.strip().lower() for x in value.split(",")]
    return value if isinstance(value, list) else []

def clean_ingredients(df):
    """Clean ingredients in DataFrame"""
    def clean_ingredient_list(ingredients):
        if not ingredients:
            return []
        return [re.sub(r's$', '', ing.strip().lower()) for ing in ingredients]
    
    df['clean_ingredients'] = df['ingredients'].apply(clean_ingredient_list)
    return df

def filter_disliked_allergies(df, disliked, allergies=None):
    """Filter out foods containing disliked or allergen ingredients"""
    # Combine disliked ingredients and allergies
    all_exclusions = set(disliked)
    if allergies:
        all_exclusions.update(allergies)
    
    if not all_exclusions:
        return df
    
    # Clean combined exclusions
    clean_exclusions = [
        re.sub(r's$', '', ing.strip().lower())
        for ing in all_exclusions
        if ing.strip() and ing.strip() != 'on'
    ]
    
    # Clean ingredients first
    df = clean_ingredients(df)
    
    # Filter out foods containing any disliked/allergen ingredient
    def contains_no_exclusions(ingredients):
        return len(set(ingredients).intersection(set(clean_exclusions))) == 0
    
    return df[df['clean_ingredients'].apply(contains_no_exclusions)]

def get_user_preferences(user_id):
    """Get user survey preferences"""
    survey_df = load_food_data(COLLECTION_SURVEY)
    user_survey = survey_df[survey_df['user_id'] == user_id]
    
    if user_survey.empty:
        raise HTTPException(400, "Survey missing")
    
    prefs = user_survey.iloc[0].get('responses', {})
    return prefs if isinstance(prefs, dict) else prefs.copy()

def calculate_ingredient_adjustments(prefs):
    """Calculate ingredient preference adjustments based on survey"""
    adjustments = defaultdict(float)
    
    # Handle different food_prefs formats
    food_prefs = prefs.get("food_preferences", {})
    
    # Convert list format to dict if needed
    if isinstance(food_prefs, list):
        # Convert from [{key: "pizza", value: "loves"}, ...] format
        food_prefs = {item["key"]: item["value"] for item in food_prefs}
    
    for dish, ingredients in DISH_INGREDIENT_MAP.items():
        # Get rating from both root and food_prefs
        raw_rating = prefs.get(dish) or food_prefs.get(dish)
        
        # Default to neutral if no rating
        numerical_rating = 3
        
        if raw_rating:
            # Handle numeric values
            if isinstance(raw_rating, (int, float)):
                numerical_rating = max(1, min(int(raw_rating), 5))
            # Handle string values
            elif isinstance(raw_rating, str):
                clean_rating = raw_rating.strip().lower()
                numerical_rating = {
                    "loves": 5, 
                    "likes": 4, 
                    "neutral": 3,
                    "dislikes": 1
                }.get(clean_rating, 3)  # Default to 3 if unknown string

        adj = RATING_ADJUSTMENTS.get(numerical_rating, 0.0)
        for ingredient in ingredients:
            adjustments[ingredient] += adj
            
    return adjustments

def prepare_tfidf_model(food_df):
    """Prepare TF-IDF model for ingredient similarity"""
    # Convert ingredient lists to strings for TF-IDF
    food_df['ingredient_text'] = food_df['ingredients'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(food_df['ingredient_text'])
    
    return vectorizer, tfidf_matrix

def ensure_json_serializable(obj):
    """Make sure all values in the object are JSON serializable"""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: ensure_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, 'to_dict'):
        return ensure_json_serializable(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        return ensure_json_serializable(obj.__dict__)
    elif pd.isna(obj):
        return None
    else:
        return str(obj)

@router.get("/recommend/")
async def recommend_foods(
    country: str = Query(..., title="Target country"), 
    diet: Optional[str] = Query(None, title="Dietary restriction"), 
    user_id: str = Depends(get_current_user), 
    top_n: int = 10
):
    try:
        print(f"Received country: {country}, user_id: {user_id}")
        
        # Load data
        food_df = load_food_data(COLLECTION_FOOD)
        
        # Select relevant columns and clean data
        food_df = food_df[['_id', 'country', 'ingredients', 'name', 'url_id']]
        food_df = food_df[food_df['ingredients'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        
        # Diet restrictions
        exclusion_ingredients = []
        if diet is not None:
            print("Diet:", diet)
            restrictions = db["special_diet"].find_one({"category": diet.lower()})
            if restrictions and '_id' in restrictions:
                restrictions['_id'] = str(restrictions['_id'])
            exclusion_ingredients = restrictions.get("restricted_ingredients", []) if restrictions else []
        else:
            print("No diet filter")
        
        # Get user preferences
        prefs = get_user_preferences(user_id)
        
        # Calculate ingredient adjustments from preferences
        ingredient_adjustments = calculate_ingredient_adjustments(prefs)
        
        # Process user comments to adjust ingredient scores
        user_comments_df = load_food_data(COLLECTION_USER_COMMENTS)
        user_comments_filtered = user_comments_df[user_comments_df['userId'] == user_id]
        
        # Join with food_df to get ingredients for each commented food
        if not user_comments_filtered.empty:
            for _, comment in user_comments_filtered.iterrows():
                food_id = comment['foodId']
                rate = comment.get('rate')
                
                # Find the corresponding food
                food_item = food_df[food_df['_id'] == food_id]
                if not food_item.empty:
                    ingredients = food_item.iloc[0]['ingredients']
                    rate_val = float(rate) if rate is not None else 3.0
                    # Scale adjustment to match survey's impact
                    adjustment = (rate_val - 3) * 1.0
                    for ingredient in ingredients:
                        ingredient_adjustments[ingredient] += adjustment
        
        # Filter to country foods and remove disliked ingredients and allergies
        country_foods = food_df[food_df['country'] == country].copy()
        country_foods = filter_disliked_allergies(
            country_foods,
            disliked=parse_prefs(prefs.get("disliked_ingredients")),
            allergies=set(parse_prefs(prefs.get("allergies"))) | set(exclusion_ingredients)
        )
        
        # Prepare TF-IDF
        vectorizer, tfidf_matrix = prepare_tfidf_model(country_foods)
        
        # Add adjustment scores based on user preferences
        def calculate_adjustment(ingredients):
            return sum(ingredient_adjustments.get(ing, 0.0) for ing in ingredients)
        
        country_foods['adjustment_score'] = country_foods['ingredients'].apply(calculate_adjustment)
        
        # Calculate TF-IDF sum score (approximation of PySpark implementation)
        # Get feature names and their index in the TF-IDF matrix
        feature_names = vectorizer.get_feature_names_out()
        feature_dict = {feature: i for i, feature in enumerate(feature_names)}
        
        def calculate_tfidf_sum(row_idx):
            row = tfidf_matrix[row_idx].toarray().flatten()
            return np.sum(row)
        
        # Calculate TF-IDF sum for each document
        tfidf_sums = []
        for i in range(tfidf_matrix.shape[0]):
            tfidf_sums.append(calculate_tfidf_sum(i))
        
        country_foods['tfidf_sum'] = tfidf_sums
        
        # Calculate total score
        country_foods['score'] = country_foods['tfidf_sum'] + country_foods['adjustment_score']
        
        # Sort by score and select top_n
        recommendations = country_foods.sort_values('score', ascending=False).head(top_n)
        
        # Find similar users recommendations
        similar_foods_json = []
        if not user_comments_filtered.empty:
            # Get foods the user rated highly
            high_rated_foods = user_comments_df[
                (user_comments_df['userId'] == user_id) & 
                (user_comments_df['rate'] >= 4)
            ]['foodId'].unique()
            
            if len(high_rated_foods) > 0:
                # Find similar users
                similar_users = user_comments_df[
                    (user_comments_df['userId'] != user_id) &
                    (user_comments_df['rate'] >= 4) &
                    (user_comments_df['foodId'].isin(high_rated_foods))
                ]['userId'].unique()
                
                if len(similar_users) > 0:
                    # Get food recommendations from similar users
                    similar_users_ratings = user_comments_df[
                        (user_comments_df['userId'].isin(similar_users)) &
                        (user_comments_df['rate'] >= 4) &
                        (~user_comments_df['foodId'].isin(high_rated_foods))
                    ]
                    
                    if not similar_users_ratings.empty:
                        # Aggregate recommendations
                        similar_food_counts = similar_users_ratings.groupby('foodId').agg(
                            similar_user_count=('userId', 'count'),
                            average_rating=('rate', 'mean')
                        ).reset_index()
                        
                        # Sort by count and rating
                        similar_food_counts = similar_food_counts.sort_values(
                            ['similar_user_count', 'average_rating'], 
                            ascending=[False, False]
                        ).head(5)
                        
                        # Join with food details
                        similar_foods = pd.merge(
                            similar_food_counts,
                            food_df[['_id', 'name', 'country', 'ingredients', 'url_id']],
                            left_on='foodId',
                            right_on='_id',
                            how='inner'
                        )
                        
                        # Filter allergies
                        if exclusion_ingredients or prefs.get('allergies'):
                            similar_foods = filter_disliked_allergies(
                                similar_foods,
                                disliked={},
                                allergies=set(parse_prefs(prefs.get("allergies"))) | set(exclusion_ingredients)
                            )
                        
                        # Convert to serializable format
                        similar_foods_json = similar_foods.to_dict('records')
                        similar_foods_json = ensure_json_serializable(similar_foods_json)
        
        # Prepare results and ensure they're JSON serializable
        main_recommendations = recommendations[[
            '_id', 'name', 'country', 'url_id', 'ingredients', 'score'
        ]].to_dict('records')
        
        # Clean up ingredients format
        for rec in main_recommendations:
            rec['ingredients'] = [ing.lower().strip() for ing in rec['ingredients']]
        
        # Ensure all data is serializable
        main_recommendations = ensure_json_serializable(main_recommendations)
        
        print("Successfully generated recommendations")
        
        return {
            "personalized_recommendations": main_recommendations,
            "similar_users_recommendations": similar_foods_json
        }

    except Exception as e:
        print(f"Error during recommendation generation: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation engine failed: {str(e)}"
        )

@router.get("/similar/{food_id}")
async def get_similar_foods(
    food_id: str, 
    country: str = Query(..., title="Target country"), 
    diet: Optional[str] = Query(None, title="Dietary restriction"), 
    user_id: str = Depends(get_current_user),
    top_n: int = 10
):
    try:
        # Load data
        food_df = load_food_data(COLLECTION_FOOD)
        
        # Select relevant columns
        food_df = food_df[['_id', 'country', 'ingredients', 'name', 'url_id']]
        food_df = food_df[food_df['ingredients'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        
        # Diet restrictions
        exclusion_ingredients = []
        if diet is not None:
            print("Diet:", diet)
            restrictions = db["special_diet"].find_one({"category": diet.lower()})
            if restrictions and '_id' in restrictions:
                restrictions['_id'] = str(restrictions['_id'])
            exclusion_ingredients = restrictions.get("restricted_ingredients", []) if restrictions else []
        else:
            print("No diet filter")
            
        # Validate target country exists
        country_count = len(food_df[food_df['country'] == country])
        if country_count == 0:
            raise HTTPException(400, detail=f"No foods available in {country}")
            
        # Find target food
        target_food = food_df[food_df['_id'] == food_id]
        if target_food.empty:
            raise HTTPException(404, detail="Food not found")
            
        target_food = target_food.iloc[0]
        
        # Get user preferences
        prefs = get_user_preferences(user_id)
        
        # Filter country foods
        country_foods = food_df[(food_df['country'] == country) & (food_df['_id'] != food_id)]
        
        # Filter allergies and dislikes
        allergies = set(parse_prefs(prefs.get("allergies")))
        if allergies or exclusion_ingredients:
            country_foods = filter_disliked_allergies(
                country_foods,
                disliked={},
                allergies=allergies | set(exclusion_ingredients)
            )
            print(f"Remaining after filters: {len(country_foods)}")
        else:
            print("No filters applied")
            
        # Convert ingredient lists to strings for TF-IDF
        food_df['ingredient_text'] = food_df['ingredients'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        country_foods['ingredient_text'] = country_foods['ingredients'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        
        # Create a combined DataFrame for fitting the vectorizer
        combined_df = pd.concat([
            pd.DataFrame([target_food]),
            country_foods
        ])
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(combined_df['ingredient_text'])
        
        # Get target food vector (first row)
        target_vector = tfidf_matrix[0]
        
        # Calculate cosine similarity with all other foods
        similarities = cosine_similarity(target_vector, tfidf_matrix[1:]).flatten()
        
        # Add similarities to country_foods
        country_foods['similarity'] = similarities
        
        # Sort by similarity and get top results
        similar_foods = country_foods.sort_values('similarity', ascending=False).head(top_n)
        
        # Apply final dietary filter if needed
        if exclusion_ingredients:
            similar_foods = filter_disliked_allergies(
                similar_foods,
                disliked=[],
                allergies=exclusion_ingredients
            )
            
        # Prepare results and ensure they're serializable
        results = similar_foods[['_id', 'name', 'country', 'ingredients', 'similarity', 'url_id']].to_dict('records')
        results = ensure_json_serializable(results)
        
        return {"results": results}
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))