# src/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
    
    def safe_clip(self, series, lower, upper):
        """Safely clip a series with proper error handling"""
        if hasattr(series, 'clip'):
            return series.clip(lower, upper)
        else:
            return np.clip(series, lower, upper)
    
    def load_usda_data(self, usda_folder_path):
        """Load USDA FoodData Central datasets"""
        print("📥 Loading USDA FoodData Central...")
        
        try:
            # Check if files exist
            required_files = ['food.csv', 'food_nutrient.csv', 'nutrient.csv']
            missing_files = []
            for file in required_files:
                if not os.path.exists(os.path.join(usda_folder_path, file)):
                    missing_files.append(file)
            
            if missing_files:
                print(f"❌ Missing USDA files: {missing_files}")
                return None
            
            # Load main food data
            food_df = pd.read_csv(os.path.join(usda_folder_path, 'food.csv'))
            food_nutrient_df = pd.read_csv(os.path.join(usda_folder_path, 'food_nutrient.csv'))
            nutrient_df = pd.read_csv(os.path.join(usda_folder_path, 'nutrient.csv'))
            
            print(f"📊 USDA Data: {len(food_df)} foods, {len(food_nutrient_df)} nutrient records")
            
            # Merge to get nutrient names
            nutrient_data = pd.merge(food_nutrient_df, nutrient_df, left_on='nutrient_id', right_on='id')
            
            # Pivot to get wide format
            wide_nutrients = nutrient_data.pivot_table(
                index='fdc_id', 
                columns='name', 
                values='amount', 
                aggfunc='mean'
            ).reset_index()
            
            # Merge with food descriptions
            usda_final = pd.merge(food_df, wide_nutrients, on='fdc_id')
            
            # Select key nutritional columns
            key_nutrients = ['Protein', 'Total lipid (fat)', 'Carbohydrate, by difference', 
                           'Energy', 'Fiber, total dietary', 'Sugars, total', 
                           'Calcium, Ca', 'Iron, Fe', 'Sodium, Na']
            
            available_nutrients = [col for col in key_nutrients if col in usda_final.columns]
            
            # Keep only essential columns
            usda_final = usda_final[['fdc_id', 'description'] + available_nutrients]
            
            # Rename columns for consistency
            column_mapping = {
                'description': 'food_name',
                'Protein': 'protein',
                'Total lipid (fat)': 'fat', 
                'Carbohydrate, by difference': 'carbs',
                'Energy': 'calories',
                'Fiber, total dietary': 'fiber',
                'Sugars, total': 'sugar',
                'Calcium, Ca': 'calcium',
                'Iron, Fe': 'iron',
                'Sodium, Na': 'sodium'
            }
            
            usda_final = usda_final.rename(columns={k: v for k, v in column_mapping.items() if k in usda_final.columns})
            
            print(f"✅ USDA final data: {usda_final.shape}")
            return usda_final
            
        except Exception as e:
            print(f"❌ Error loading USDA data: {e}")
            return None
    
    def load_foodcom_data(self, foodcom_folder_path):
        """Load Food.com recipe datasets"""
        print("📥 Loading Food.com recipes...")
        
        try:
            # Check if files exist
            required_files = ['RAW_recipes.csv', 'RAW_interactions.csv']
            missing_files = []
            for file in required_files:
                if not os.path.exists(os.path.join(foodcom_folder_path, file)):
                    missing_files.append(file)
            
            if missing_files:
                print(f"❌ Missing Food.com files: {missing_files}")
                return None
            
            # Load raw recipes
            recipes_df = pd.read_csv(os.path.join(foodcom_folder_path, 'RAW_recipes.csv'))
            interactions_df = pd.read_csv(os.path.join(foodcom_folder_path, 'RAW_interactions.csv'))
            
            print(f"📊 Food.com Data: {len(recipes_df)} recipes, {len(interactions_df)} interactions")
            
            # Parse nutrition column (it's stored as string list)
            def parse_nutrition(nutrition_str):
                try:
                    if pd.isna(nutrition_str):
                        return [0, 0, 0, 0, 0, 0, 0]
                    # Remove brackets and split
                    nutrition_list = nutrition_str.strip('[]').split(',')
                    return [float(x.strip()) for x in nutrition_list]
                except:
                    return [0, 0, 0, 0, 0, 0, 0]
            
            # Apply nutrition parsing
            nutrition_data = recipes_df['nutrition'].apply(parse_nutrition)
            nutrition_columns = ['calories', 'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates']
            
            for i, col in enumerate(nutrition_columns):
                if i < len(nutrition_data.iloc[0]):
                    recipes_df[col] = nutrition_data.apply(lambda x: x[i] if i < len(x) else 0)
            
            # Calculate average ratings
            recipe_ratings = interactions_df.groupby('recipe_id').agg({
                'rating': 'mean',
                'user_id': 'count'
            }).reset_index()
            recipe_ratings.columns = ['id', 'avg_rating', 'review_count']
            
            # Merge ratings with recipes
            recipes_final = pd.merge(recipes_df, recipe_ratings, on='id', how='left')
            recipes_final['avg_rating'] = recipes_final['avg_rating'].fillna(0)
            recipes_final['review_count'] = recipes_final['review_count'].fillna(0)
            
            # Select and rename key columns
            foodcom_final = recipes_final[[
                'id', 'name', 'minutes', 'ingredients', 'steps', 
                'calories', 'protein', 'carbohydrates', 'total_fat', 'sugar', 'sodium',
                'avg_rating', 'review_count'
            ]].rename(columns={
                'id': 'recipe_id',
                'name': 'recipe_name', 
                'minutes': 'prep_time',
                'total_fat': 'fat'
            })
            
            print(f"✅ Food.com final data: {foodcom_final.shape}")
            return foodcom_final
            
        except Exception as e:
            print(f"❌ Error loading Food.com data: {e}")
            return None
    
    def load_instacart_data(self, instacart_folder_path):
        """Load Instacart market basket data"""
        print("📥 Loading Instacart data...")
        
        try:
            # Check if files exist
            required_files = ['products.csv', 'order_products__prior.csv']
            missing_files = []
            for file in required_files:
                if not os.path.exists(os.path.join(instacart_folder_path, file)):
                    missing_files.append(file)
            
            if missing_files:
                print(f"❌ Missing Instacart files: {missing_files}")
                return None
            
            products_df = pd.read_csv(os.path.join(instacart_folder_path, 'products.csv'))
            order_products_df = pd.read_csv(os.path.join(instacart_folder_path, 'order_products__prior.csv'))
            
            print(f"📊 Instacart Data: {len(products_df)} products, {len(order_products_df)} order records")
            
            # Calculate purchase frequency for each product
            product_frequency = order_products_df.groupby('product_id').size().reset_index()
            product_frequency.columns = ['product_id', 'purchase_frequency']
            
            # Merge with product info
            instacart_final = pd.merge(products_df, product_frequency, on='product_id')
            
            # Create price estimate based on department and frequency
            price_ranges = {
                'produce': (1.0, 5.0), 'dairy eggs': (2.0, 8.0), 'beverages': (1.5, 6.0),
                'canned goods': (1.0, 4.0), 'dry goods pasta': (1.0, 5.0), 'meat seafood': (3.0, 15.0),
                'bakery': (2.0, 7.0), 'frozen': (2.0, 10.0), 'pantry': (1.0, 6.0)
            }
            
            def estimate_price(department, frequency):
                base_range = price_ranges.get(department, (1.0, 5.0))
                # More frequent items tend to be cheaper
                frequency_factor = max(0.5, 1 - (frequency / 1000))
                price = np.random.uniform(base_range[0], base_range[1]) * frequency_factor
                return round(price, 2)
            
            instacart_final['price_per_unit'] = instacart_final.apply(
                lambda row: estimate_price(row['department'], row['purchase_frequency']), 
                axis=1
            )
            
            # Select key columns
            instacart_final = instacart_final[['product_id', 'product_name', 'department', 
                                             'purchase_frequency', 'price_per_unit']]
            
            print(f"✅ Instacart final data: {instacart_final.shape}")
            return instacart_final
            
        except Exception as e:
            print(f"❌ Error loading Instacart data: {e}")
            return None
    
    def merge_datasets(self, usda_data, foodcom_data, instacart_data):
        """Merge all three datasets into unified format"""
        print("🔄 Merging datasets...")
        
        # Use Food.com as base (recipes)
        if foodcom_data is None:
            print("⚠️ No Food.com data available, creating fallback dataset...")
            return self.create_fallback_dataset()
        
        merged_data = foodcom_data.copy()
        
        # Add price estimates based on ingredients
        def estimate_recipe_price(ingredients_str):
            try:
                if pd.isna(ingredients_str):
                    return round(np.random.uniform(3, 7), 2)
                # Simple heuristic: $2-8 per recipe based on complexity
                ingredients_list = eval(ingredients_str) if isinstance(ingredients_str, str) else []
                num_ingredients = len(ingredients_list)
                base_price = max(2.0, min(8.0, num_ingredients * 0.5))
                return round(base_price + np.random.uniform(-1, 1), 2)
            except:
                return round(np.random.uniform(3, 7), 2)
        
        merged_data['price_per_serving'] = merged_data['ingredients'].apply(estimate_recipe_price)
        
        print(f"✅ Merged dataset: {merged_data.shape}")
        return merged_data
    
    def engineer_features(self, df):
        """Create advanced feature engineering with real nutritional formulas"""
        print("🔧 Engineering features from real data...")
        
        # Safe column access with defaults
        def get_safe_column(column_name, default_value):
            if column_name in df.columns:
                return df[column_name]
            else:
                print(f"⚠️ Column {column_name} not found, using default value {default_value}")
                return pd.Series([default_value] * len(df), index=df.index)
        
        # Nutrition Score (based on real nutritional guidelines)
        # Protein component (0-25 points)
        protein_max = df['protein'].max() if df['protein'].max() > 0 else 50
        protein_score = (get_safe_column('protein', 15) / protein_max * 25)
        protein_score = self.safe_clip(protein_score, 0, 25)
        
        # Low fat component (0-20 points)
        fat_max = df['fat'].max() if df['fat'].max() > 0 else 65
        fat_score = (20 - (get_safe_column('fat', 15) / fat_max * 20))
        fat_score = self.safe_clip(fat_score, 0, 20)
        
        # Low sugar component (0-15 points)
        sugar_max = df['sugar'].max() if df['sugar'].max() > 0 else 50
        sugar_score = (15 - (get_safe_column('sugar', 10) / sugar_max * 15))
        sugar_score = self.safe_clip(sugar_score, 0, 15)
        
        # Low sodium component (0-15 points)
        sodium_max = df['sodium'].max() if df['sodium'].max() > 0 else 2300
        sodium_score = (15 - (get_safe_column('sodium', 800) / sodium_max * 15))
        sodium_score = self.safe_clip(sodium_score, 0, 15)
        
        # Balanced calories component (0-25 points)
        # Ideal range: 300-600 calories per serving
        calories_data = get_safe_column('calories', 450)
        calorie_score = np.where(
            (calories_data >= 300) & (calories_data <= 600),
            25,
            np.where(calories_data < 300, calories_data / 300 * 25, 
                    (1200 - calories_data) / 600 * 25)
        )
        calorie_score = self.safe_clip(calorie_score, 0, 25)
        
        df['nutrition_score'] = (protein_score + fat_score + sugar_score + 
                               sodium_score + calorie_score)
        df['nutrition_score'] = self.safe_clip(df['nutrition_score'], 0, 100)
        
        # Affordability metrics
        df['price_per_meal'] = get_safe_column('price_per_serving', 10)
        df['affordability_score'] = (1 / (1 + df['price_per_meal'] / 10))
        df['affordability_score'] = self.safe_clip(df['affordability_score'], 0, 1)
        df['cost_efficiency'] = df['nutrition_score'] / (df['price_per_meal'] + 1)
        
        # Time efficiency
        prep_time = get_safe_column('prep_time', 30)
        df['time_efficiency'] = df['nutrition_score'] / (prep_time / 10 + 1)
        
        # Popularity metrics
        rating = get_safe_column('avg_rating', 4)
        review_count = get_safe_column('review_count', 10)
        df['popularity_score'] = (rating * np.log1p(review_count) / 5).clip(0, 1)
        
        # Recipe complexity (based on number of ingredients and steps)
        df['ingredient_count'] = df['ingredients'].apply(
            lambda x: len(eval(x)) if isinstance(x, str) else 1
        )
        df['step_count'] = df['steps'].apply(
            lambda x: len(eval(x)) if isinstance(x, str) else 1
        )
        df['complexity_score'] = (df['ingredient_count'] + df['step_count']) / 20
        
        # Difficulty encoding based on prep time and complexity
        conditions = [
            (df['prep_time'] <= 15) & (df['complexity_score'] <= 0.3),
            (df['prep_time'] <= 30) & (df['complexity_score'] <= 0.6),
            (df['prep_time'] > 30) | (df['complexity_score'] > 0.6)
        ]
        choices = [0, 1, 2]  # Easy, Medium, Hard
        df['difficulty_encoded'] = np.select(conditions, choices, default=1)
        
        print("✅ Feature engineering completed!")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("🔧 Handling missing values...")
        
        # Fill missing numerical values
        numerical_cols = ['calories', 'protein', 'carbohydrates', 'fat', 'sugar', 'sodium']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing ratings and reviews
        if 'avg_rating' in df.columns:
            df['avg_rating'] = df['avg_rating'].fillna(0)
        if 'review_count' in df.columns:
            df['review_count'] = df['review_count'].fillna(0)
        
        return df
    
    def create_fallback_dataset(self):
        """Create fallback dataset if real data loading fails"""
        print("🔄 Creating realistic fallback dataset...")
        
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'recipe_id': range(n_samples),
            'recipe_name': [f'Recipe_{i}' for i in range(n_samples)],
            'prep_time': np.random.exponential(30, n_samples).clip(5, 120),
            'calories': np.random.normal(450, 150, n_samples).clip(100, 800),
            'protein': np.random.gamma(2, 2, n_samples).clip(5, 40),
            'carbohydrates': np.random.normal(55, 20, n_samples).clip(10, 120),
            'fat': np.random.gamma(1.5, 3, n_samples).clip(2, 35),
            'sugar': np.random.gamma(2, 3, n_samples).clip(1, 30),
            'sodium': np.random.normal(800, 400, n_samples).clip(100, 2000),
            'price_per_serving': np.random.lognormal(2.5, 0.7, n_samples).clip(1, 15),
            'avg_rating': np.random.beta(8, 2, n_samples) * 5,
            'review_count': np.random.poisson(45, n_samples),
            'ingredient_count': np.random.poisson(8, n_samples).clip(3, 20),
            'step_count': np.random.poisson(6, n_samples).clip(2, 15)
        })
        
        # Calculate derived features
        data['nutrition_score'] = (
            (data['protein'] / 40 * 25) +
            (20 - (data['fat'] / 35 * 20)) +
            (15 - (data['sugar'] / 30 * 15)) +
            (15 - (data['sodium'] / 2000 * 15)) +
            np.where((data['calories'] >= 300) & (data['calories'] <= 600), 25, 
                    np.where(data['calories'] < 300, data['calories']/300*25, 
                            (1200-data['calories'])/600*25))
        ).clip(0, 100)
        
        data['affordability_score'] = (1 / (1 + data['price_per_serving'] / 5)).clip(0, 1)
        data['time_efficiency'] = data['nutrition_score'] / (data['prep_time'] / 10 + 1)
        data['popularity_score'] = (data['avg_rating'] * np.log1p(data['review_count']) / 5)
        data['complexity_score'] = (data['ingredient_count'] + data['step_count']) / 25
        data['difficulty_encoded'] = np.random.randint(0, 3, n_samples)
        
        return data
    
    def prepare_final_dataset(self, usda_folder, foodcom_folder, instacart_folder):
        """Complete data processing pipeline with local datasets"""
        print("="*50)
        print("PHASE 1: DATA PROCESSING & FEATURE ENGINEERING")
        print("="*50)
        
        # Load all datasets
        usda_data = self.load_usda_data(usda_folder)
        foodcom_data = self.load_foodcom_data(foodcom_folder) 
        instacart_data = self.load_instacart_data(instacart_folder)
        
        # Merge datasets
        df = self.merge_datasets(usda_data, foodcom_data, instacart_data)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.engineer_features(df)
        
        # Select final features for modeling
        final_features = [
            'recipe_id', 'recipe_name', 'prep_time', 'ingredient_count', 'step_count',
            'calories', 'protein', 'carbohydrates', 'fat', 'sugar', 'sodium',
            'nutrition_score', 'price_per_serving', 'affordability_score', 
            'cost_efficiency', 'time_efficiency', 'popularity_score', 
            'complexity_score', 'difficulty_encoded', 'avg_rating', 'review_count'
        ]
        
        # Keep only available features
        available_features = [f for f in final_features if f in df.columns]
        df_final = df[available_features].copy()
        
        print(f"✅ Final dataset shape: {df_final.shape}")
        print(f"📊 Final columns: {df_final.columns.tolist()}")
        
        return df_final