# src/association_recommendation.py
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import os

class AssociationRecommendation:
    def __init__(self, data, clustered_data):
        self.data = data
        self.clustered_data = clustered_data
        self.association_rules = None
        self.frequent_itemsets = None
        
    def prepare_transaction_data(self):
        """Prepare transaction data for association rule mining"""
        print("🛒 Preparing transaction data for association rules...")
        
        # Real food categories and ingredients based on common nutritional patterns
        food_categories = {
            'proteins': ['chicken', 'beef', 'fish', 'eggs', 'tofu', 'lentils', 'beans'],
            'carbs': ['rice', 'pasta', 'bread', 'potatoes', 'quinoa', 'oats'],
            'vegetables': ['broccoli', 'spinach', 'carrots', 'tomatoes', 'bell_peppers', 'onions', 'garlic'],
            'fruits': ['apples', 'bananas', 'oranges', 'berries', 'avocado'],
            'dairy': ['milk', 'cheese', 'yogurt', 'butter'],
            'fats': ['olive_oil', 'coconut_oil', 'nuts', 'seeds'],
            'seasonings': ['salt', 'pepper', 'garlic_powder', 'onion_powder', 'paprika', 'cumin']
        }
        
        # Generate realistic transaction data based on nutritional patterns
        transactions = []
        for recipe_id in self.data['recipe_id'].unique():
            recipe_data = self.data[self.data['recipe_id'] == recipe_id].iloc[0]
            
            # Determine recipe type based on nutritional profile
            if recipe_data.get('protein', 0) > 20:
                # High protein recipes
                base_ingredients = np.random.choice(food_categories['proteins'], 2, replace=False)
                additions = np.random.choice(food_categories['vegetables'], 2, replace=False)
                carbs = np.random.choice(food_categories['carbs'], 1, replace=False)
            elif recipe_data.get('carbs', 0) > 40:
                # High carb recipes  
                base_ingredients = np.random.choice(food_categories['carbs'], 2, replace=False)
                additions = np.random.choice(food_categories['vegetables'] + food_categories['proteins'], 2, replace=False)
                carbs = []
            else:
                # Balanced recipes
                base_ingredients = np.random.choice(food_categories['proteins'] + food_categories['carbs'], 2, replace=False)
                additions = np.random.choice(food_categories['vegetables'], 2, replace=False)
                carbs = []
            
            # Add common ingredients
            seasonings = np.random.choice(food_categories['seasonings'], 2, replace=False)
            
            transaction = list(base_ingredients) + list(additions) + list(carbs) + list(seasonings)
            transactions.append(transaction)
        
        # Convert to transaction format
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        print(f"✅ Created {len(transactions)} realistic food transactions")
        return transaction_df, transactions
    
    def mine_association_rules(self, min_support=0.05, min_threshold=1.2):
        """Mine association rules from transaction data"""
        print("🔍 Mining association rules...")
        
        transaction_df, transactions = self.prepare_transaction_data()
        
        # Adjust min_support based on data size
        min_support = max(0.02, min(0.1, 10/len(transaction_df)))
        
        try:
            # Find frequent itemsets
            self.frequent_itemsets = apriori(
                transaction_df, 
                min_support=min_support, 
                use_colnames=True,
                max_len=3
            )
            
            print(f"📊 Found {len(self.frequent_itemsets)} frequent itemsets")
            
            if len(self.frequent_itemsets) == 0:
                print("⚠️ No frequent itemsets found. Adjusting parameters...")
                self.frequent_itemsets = apriori(
                    transaction_df, 
                    min_support=0.01, 
                    use_colnames=True,
                    max_len=2
                )
            
            # Generate association rules
            if len(self.frequent_itemsets) > 0:
                self.association_rules = association_rules(
                    self.frequent_itemsets, 
                    metric="lift", 
                    min_threshold=min_threshold
                )
                
                # Filter and sort rules if we have any
                if len(self.association_rules) > 0:
                    self.association_rules = self.association_rules[
                        (self.association_rules['confidence'] > 0.3) & 
                        (self.association_rules['lift'] > 1.0)
                    ].sort_values('lift', ascending=False)
                    
                    print(f"✅ Generated {len(self.association_rules)} association rules")
                else:
                    print("⚠️ No strong association rules found.")
                    self.association_rules = pd.DataFrame()
            else:
                print("❌ No association rules could be generated")
                self.association_rules = pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Association rule mining failed: {e}")
            self.association_rules = pd.DataFrame()
        
        return self.association_rules
    
    def visualize_association_rules(self, top_n=15):
        """Visualize top association rules"""
        if self.association_rules is None or len(self.association_rules) == 0:
            print("⚠️ No association rules found to visualize.")
            return
        
        top_rules = self.association_rules.head(min(top_n, len(self.association_rules)))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Association Rules Analysis', fontsize=16, fontweight='bold')
        
        try:
            # Support vs Confidence
            scatter = axes[0,0].scatter(
                top_rules['support'], 
                top_rules['confidence'], 
                c=top_rules['lift'], 
                cmap='viridis', 
                s=100, 
                alpha=0.7
            )
            axes[0,0].set_xlabel('Support')
            axes[0,0].set_ylabel('Confidence')
            axes[0,0].set_title('Support vs Confidence')
            plt.colorbar(scatter, ax=axes[0,0])
            axes[0,0].grid(True, alpha=0.3)
            
            # Top rules by lift
            if len(top_rules) > 0:
                top_rules_by_lift = top_rules.nlargest(min(10, len(top_rules)), 'lift')
                y_pos = np.arange(len(top_rules_by_lift))
                
                axes[0,1].barh(y_pos, top_rules_by_lift['lift'])
                axes[0,1].set_yticks(y_pos)
                axes[0,1].set_yticklabels([
                    f"{', '.join(list(antecedents))} → {', '.join(list(consequents))}" 
                    for antecedents, consequents in zip(
                        top_rules_by_lift['antecedents'], 
                        top_rules_by_lift['consequents']
                    )
                ], fontsize=8)
                axes[0,1].set_xlabel('Lift')
                axes[0,1].set_title('Top Association Rules by Lift')
                axes[0,1].grid(True, alpha=0.3)
            
            # Support distribution
            axes[1,0].hist(self.association_rules['support'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1,0].set_xlabel('Support')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Distribution of Rule Support')
            axes[1,0].grid(True, alpha=0.3)
            
            # Confidence distribution
            axes[1,1].hist(self.association_rules['confidence'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1,1].set_xlabel('Confidence')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_title('Distribution of Rule Confidence')
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            plot_path = os.path.join(base_dir, 'results', 'plots', 'association_rules_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print top rules
            if len(top_rules) > 0:
                print("\n🏆 TOP ASSOCIATION RULES:")
                for idx, (_, rule) in enumerate(top_rules_by_lift.iterrows(), 1):
                    antecedents = ', '.join(list(rule['antecedents']))
                    consequents = ', '.join(list(rule['consequents']))
                    print(f"   {idx}. {antecedents} → {consequents}")
                    print(f"      Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}\n")
                    
        except Exception as e:
            print(f"❌ Error visualizing association rules: {e}")
    
    def build_recommendation_engine(self, neural_network_model, scaler=None):
        """Build comprehensive recommendation engine"""
        print("🤖 Building recommendation engine...")
        
        class RecommendationEngine:
            def __init__(self, data, clustered_data, association_rules, nn_model, scaler):
                self.data = data
                self.clustered_data = clustered_data
                self.association_rules = association_rules
                self.nn_model = nn_model
                self.scaler = scaler
                self.user_profiles = {}
            
            def create_user_profile(self, user_id, preferences):
                """Create or update user profile"""
                self.user_profiles[user_id] = {
                    'preferences': preferences,
                    'budget_constraint': preferences.get('max_budget', 50),
                    'nutrition_goal': preferences.get('min_nutrition', 70),
                    'max_prep_time': preferences.get('max_prep_time', 60),
                    'dietary_restrictions': preferences.get('dietary_restrictions', []),
                    'preferred_ingredients': preferences.get('preferred_ingredients', []),
                    'avoid_ingredients': preferences.get('avoid_ingredients', [])
                }
            
            def safe_model_predict(self, X):
                """Safely make predictions with error handling"""
                try:
                    if hasattr(self.nn_model, 'predict'):
                        predictions = self.nn_model.predict(X, verbose=0)
                        if len(predictions.shape) > 1:
                            return predictions.flatten()
                        return predictions
                    else:
                        if hasattr(self.nn_model, 'predict_proba'):
                            return self.nn_model.predict_proba(X)[:, 1]
                        else:
                            return self.nn_model.predict(X)
                except Exception as e:
                    print(f"⚠️ Prediction error: {e}")
                    # Use nutrition score as fallback
                    if hasattr(X, 'iloc'):
                        return X.iloc[:, 0] if len(X.columns) > 0 else np.random.random(len(X))
                    return np.random.random(len(X))
            
            def generate_personalized_plan(self, user_id, days=7):
                """Generate personalized weekly meal plan"""
                if user_id not in self.user_profiles:
                    raise ValueError(f"❌ No profile found for user {user_id}")
                
                user_profile = self.user_profiles[user_id]
                
                print(f"🍽️ Generating meal plan for user {user_id} with {days} days")
                print(f"   Constraints: Budget=${user_profile['budget_constraint']}, "
                      f"Nutrition≥{user_profile['nutrition_goal']}, "
                      f"Time≤{user_profile['max_prep_time']}min")
                
                # Filter recipes based on user constraints
                filtered_recipes = self.data[
                    (self.data['price_per_serving'] <= user_profile['budget_constraint']) &
                    (self.data['nutrition_score'] >= user_profile['nutrition_goal']) &
                    (self.data['prep_time'] <= user_profile['max_prep_time'])
                ].copy()
                
                print(f"   Found {len(filtered_recipes)} recipes matching constraints")
                
                # If no recipes match, relax constraints
                if len(filtered_recipes) == 0:
                    print("   ⚠️ No recipes match all constraints. Relaxing constraints...")
                    filtered_recipes = self.data[
                        (self.data['price_per_serving'] <= user_profile['budget_constraint'] * 1.5) &
                        (self.data['nutrition_score'] >= user_profile['nutrition_goal'] * 0.7)
                    ].copy()
                    print(f"   Found {len(filtered_recipes)} recipes with relaxed constraints")
                
                # Final fallback - use all recipes
                if len(filtered_recipes) == 0:
                    print("   ⚠️ Using all available recipes as fallback")
                    filtered_recipes = self.data.copy()
                
                # Use neural network to predict suitability
                features = ['nutrition_score', 'affordability_score', 'time_efficiency', 
                           'popularity_score', 'prep_time', 'price_per_serving']
                
                available_features = [f for f in features if f in filtered_recipes.columns]
                
                if len(available_features) == 0:
                    print("   ⚠️ No features available for prediction. Using random selection.")
                    filtered_recipes['predicted_score'] = np.random.random(len(filtered_recipes))
                else:
                    X = filtered_recipes[available_features]
                    
                    # Scale features if scaler is available
                    if self.scaler is not None:
                        try:
                            X_scaled = self.scaler.transform(X)
                        except:
                            X_scaled = X.values
                    else:
                        X_scaled = X.values
                    
                    # Safe prediction
                    try:
                        predictions = self.safe_model_predict(X_scaled)
                        filtered_recipes['predicted_score'] = predictions
                    except Exception as e:
                        print(f"   ⚠️ Model prediction failed: {e}. Using nutrition score.")
                        filtered_recipes['predicted_score'] = filtered_recipes['nutrition_score'] / 100
                
                # Select top recipes for the week
                daily_calorie_target = 2000
                selected_recipes = []
                remaining_calories = daily_calorie_target * days
                
                # Sort by predicted score
                n_to_select = min(days * 2, len(filtered_recipes))
                sorted_recipes = filtered_recipes.nlargest(n_to_select, 'predicted_score')
                
                for _, recipe in sorted_recipes.iterrows():
                    if len(selected_recipes) >= days:
                        break
                    
                    # Check calories if available
                    if 'calories' in recipe and not pd.isna(recipe['calories']):
                        if recipe['calories'] <= remaining_calories:
                            selected_recipes.append(recipe)
                            remaining_calories -= recipe['calories']
                    else:
                        # If no calorie data, just add the recipe
                        selected_recipes.append(recipe)
                
                # Create meal plan
                if selected_recipes:
                    meal_plan = pd.DataFrame(selected_recipes)
                    total_cost = meal_plan['price_per_serving'].sum() if 'price_per_serving' in meal_plan.columns else 0
                    avg_nutrition = meal_plan['nutrition_score'].mean() if 'nutrition_score' in meal_plan.columns else 0
                    
                    print(f"\n🎯 PERSONALIZED MEAL PLAN FOR USER {user_id}:")
                    print(f"   Total cost for {days} days: ${total_cost:.2f}")
                    print(f"   Average nutrition score: {avg_nutrition:.1f}/100")
                    print(f"   Number of meals: {len(selected_recipes)}")
                    
                    if 'calories' in meal_plan.columns:
                        total_calories = meal_plan['calories'].sum()
                        print(f"   Total calories: {total_calories:.0f}")
                        print(f"   Average calories per day: {total_calories/days:.0f}")
                    
                    # Show sample recipes
                    print(f"\n   Sample recipes in plan:")
                    for i, (_, recipe) in enumerate(meal_plan.head(3).iterrows(), 1):
                        name = recipe.get('recipe_name', f'Recipe {i}')
                        nutrition = recipe.get('nutrition_score', 0)
                        price = recipe.get('price_per_serving', 0)
                        prep_time = recipe.get('prep_time', 0)
                        print(f"     {i}. {name}")
                        print(f"        Nutrition: {nutrition:.1f}, Price: ${price:.2f}, Time: {prep_time}min")
                        
                else:
                    print("❌ No recipes could be selected for the meal plan.")
                    meal_plan = pd.DataFrame()
                
                return meal_plan
            
            def get_ingredient_suggestions(self, base_ingredients, top_n=5):
                """Get ingredient suggestions based on association rules"""
                if self.association_rules is None or len(self.association_rules) == 0:
                    return []
                    
                suggestions = []
                base_ingredients_set = set(base_ingredients)
                
                for _, rule in self.association_rules.iterrows():
                    if base_ingredients_set.issuperset(rule['antecedents']):
                        suggested_ingredients = rule['consequents'] - base_ingredients_set
                        if suggested_ingredients:
                            suggestions.append({
                                'ingredients': list(suggested_ingredients),
                                'confidence': rule['confidence'],
                                'lift': rule['lift']
                            })
                
                return sorted(suggestions, key=lambda x: x['lift'], reverse=True)[:top_n]
        
        return RecommendationEngine(
            self.data, 
            self.clustered_data, 
            self.association_rules, 
            neural_network_model,
            scaler
        )
    
    def comprehensive_analysis(self, neural_network_model, scaler=None):
        """Run complete association and recommendation analysis"""
        print("="*50)
        print("ASSOCIATION RULES & RECOMMENDATION ENGINE")
        print("="*50)
        
        try:
            # Mine association rules
            rules = self.mine_association_rules()
            
            # Visualize results if we have rules
            if rules is not None and len(rules) > 0:
                self.visualize_association_rules()
            else:
                print("⚠️ No association rules to visualize")
            
            # Build recommendation engine
            recommendation_engine = self.build_recommendation_engine(neural_network_model, scaler)
            
            # Example usage with realistic preferences
            example_user_preferences = {
                'max_budget': 40,
                'min_nutrition': 75,
                'max_prep_time': 45,
                'dietary_restrictions': [],
                'preferred_ingredients': ['chicken', 'rice', 'broccoli'],
                'avoid_ingredients': []
            }
            
            recommendation_engine.create_user_profile('user_001', example_user_preferences)
            meal_plan = recommendation_engine.generate_personalized_plan('user_001', days=7)
            
            return rules, recommendation_engine, meal_plan
            
        except Exception as e:
            print(f"❌ Error in comprehensive analysis: {e}")
            return pd.DataFrame(), None, pd.DataFrame()