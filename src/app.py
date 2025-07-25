import streamlit as st
import re
from recommender import CocktailRecommender

# Page config
st.set_page_config(
    page_title="🍹 Cocktail Suggestions",
    page_icon="🍹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cocktail-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .similarity-score {
        background: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .ingredient-tag {
        background: #FF9800;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_recommender():
    """Initialize the cocktail recommender"""
    return CocktailRecommender()

def clean_value(value):
    """Clean potential XML tags from string values"""
    if value is None:
        return ""
    return re.sub(r'</?\w+>', '', str(value)).strip()

def display_cocktail(cocktail):
    """Display a cocktail in a nice card format"""
    # Clean all string fields
    cleaned = {
        'name': clean_value(cocktail.get('name')),
        'category': clean_value(cocktail.get('category')),
        'alcoholic': clean_value(cocktail.get('alcoholic')),
        'glass': clean_value(cocktail.get('glass')),
        'ingredients': clean_value(cocktail.get('ingredients')),
        'recipe': clean_value(cocktail.get('recipe')),
        'similarity': cocktail.get('similarity')
    }

    similarity = cleaned['similarity']
    if similarity is not None:
        if isinstance(similarity, (int, float)):
            sim_percent = int(similarity * 100) if similarity <= 1 else int(similarity)
        else:
            sim_percent = similarity
        sim_html = f'<div class="similarity-score">Match: {sim_percent}%</div>'
    else:
        sim_html = ''

    ingredients_html = ''
    if cleaned['ingredients']:
        ingredients = [ing.strip() for ing in cleaned['ingredients'].split(',') if ing.strip()]
        ingredients_html = '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">' + ''.join(f'<span class="ingredient-tag">{ing}</span>' for ing in ingredients[:8]) + '</div>'

    st.markdown(f"""
    <div class="cocktail-card">
        <h3>🍹 {cleaned['name']}</h3>
        {sim_html}
        <p><strong>Category:</strong> {cleaned['category']}</p>
        <p><strong>Type:</strong> {cleaned['alcoholic']}</p>
        <p><strong>Glass:</strong> {cleaned['glass']}</p>
        <p><strong>Ingredients:</strong></p>
        {ingredients_html}
    </div>
    """, unsafe_allow_html=True)
    
    # Recipe in expander
    with st.expander("📖 View Recipe", expanded=False):
        st.text(cleaned['recipe'])

def handle_name_search(recommender):
    st.subheader("Search Cocktails by Name")
    cocktail_name = st.text_input("Enter cocktail name:", placeholder="e.g., Margarita, Mojito")
    
    if cocktail_name:
        with st.spinner("Searching..."):
            return recommender.get_cocktail_by_name(cocktail_name)
    return []

def handle_ingredients_search(recommender, common_ingredients):
    st.subheader("Find Cocktails by Ingredients")
    
    col_a, col_b = st.columns(2)
    with col_a:
        selected_common = st.multiselect("Quick select:", common_ingredients)
    with col_b:
        custom_ingredients = st.text_input("Add custom ingredients (comma-separated):")
    
    all_ingredients = selected_common.copy()
    if custom_ingredients:
        all_ingredients.extend([ing.strip() for ing in custom_ingredients.split(',') if ing.strip()])
    
    if all_ingredients:
        st.write("Selected ingredients:", ", ".join(all_ingredients))
        if st.button("Find Cocktails", type="primary", key="ingredients_search"):
            with st.spinner("Finding perfect matches..."):
                return recommender.recommend_by_ingredients(all_ingredients, limit=10)
    return []

def handle_style_search(recommender):
    st.subheader("Find Cocktails by Style")
    
    style_options = [
        "sweet", "sour", "bitter", "strong", "light", "fruity", 
        "creamy", "refreshing", "exotic", "classic", "tropical"
    ]
    
    selected_styles = st.multiselect("What mood are you in?", style_options)
    
    if selected_styles:
        if st.button("Find Cocktails", type="primary", key="style_search"):
            with st.spinner("Finding your mood..."):
                return recommender.recommend_by_style(selected_styles, limit=10)
    return []

def handle_occasion_search(recommender):
    st.subheader("Find Cocktails for Your Occasion")
    
    occasion = st.selectbox("What's the occasion?", [
        "", "party", "date night", "summer evening", "winter warmer",
        "brunch", "after dinner", "celebration", "relaxing at home"
    ])
    
    if occasion:
        if st.button("Find Cocktails", type="primary", key="occasion_search"):
            with st.spinner("Planning your perfect drink..."):
                return recommender.recommend_by_occasion(occasion, limit=10)
    return []

def handle_mixed_search(recommender, common_ingredients, alcoholic_options):
    st.subheader("Customize Your Perfect Search")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        ingredients = st.multiselect("Preferred ingredients:", common_ingredients)
        styles = st.multiselect("Style preferences:", [
            "sweet", "sour", "strong", "light", "fruity", "refreshing"
        ])
    
    with col_b:
        occasion = st.selectbox("Occasion:", [
            "", "party", "date night", "summer", "winter", "brunch"
        ])
        alcoholic_pref = st.selectbox("Alcoholic preference:", [""] + alcoholic_options)
    
    if any([ingredients, styles, occasion, alcoholic_pref]):
        if st.button("Find My Perfect Cocktail", type="primary", key="mixed_search"):
            with st.spinner("Analyzing your preferences..."):
                return recommender.recommend_by_mixed_preferences(
                    ingredients=ingredients or None,
                    style=styles or None,
                    occasion=occasion or None,
                    alcoholic_preference=alcoholic_pref or None,
                    limit=10
                )
    return []

def handle_category_search(recommender):
    st.subheader("Browse by Category")
    
    category = st.selectbox("Choose a category:", [
        "", "Ordinary Drink", "Cocktail", "Shot", "Coffee / Tea",
        "Homemade Liqueur", "Punch / Party Drink", "Beer", "Soft Drink"
    ])
    
    if category:
        with st.spinner("Loading category..."):
            return recommender.get_cocktails_by_category(category, limit=10)
    return []

def handle_random_search(recommender):
    st.subheader("Discover Something New!")
    st.write("Let AI surprise you with random cocktail suggestions!")
    
    if st.button("🎲 Surprise Me!", type="primary", key="random_search"):
        with st.spinner("Rolling the dice..."):
            return recommender.get_random_cocktails(limit=6)
    return []

def main():
    # Header
    st.markdown('<h1 class="main-header">🍹 AI-Powered Cocktail Suggestions</h1>', unsafe_allow_html=True)
    st.markdown("### Discover your perfect cocktail using AI and vector similarity!")
    
    # Initialize session state for results
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'last_search_type' not in st.session_state:
        st.session_state.last_search_type = ""
    
    # Initialize recommender
    try:
        recommender = get_recommender()
    except Exception as e:
        st.error(f"Error initializing recommender: {e}")
        st.info("Make sure your database is set up and the environment variables are configured.")
        return
    
    # Sidebar for filters and preferences
    with st.sidebar:
        st.header("🎯 Your Preferences")
        
        search_type = st.selectbox(
            "How would you like to find cocktails?",
            [
                "🥃 By Ingredients",
                "🎭 By Style/Mood",
                "🎉 By Occasion",
                "🎲 Mixed Preferences",
                "📂 By Category",
                "🎰 Random Discovery",
                "🔍 Search by Name"
            ]
        )
        
        st.divider()
        
        # Common ingredients for quick selection
        common_ingredients = [
            "vodka", "gin", "rum", "whiskey", "tequila", "bourbon",
            "lime", "lemon", "orange", "cranberry", "pineapple",
            "mint", "basil", "simple syrup", "triple sec", "vermouth"
        ]
        
        alcoholic_options = ["Alcoholic", "Non alcoholic", "Optional alcohol"]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Clear results if search type changed
        if st.session_state.last_search_type != search_type:
            st.session_state.search_results = []
            st.session_state.last_search_type = search_type
        
        # Handle different search types
        if search_type == "🥃 By Ingredients":
            results = handle_ingredients_search(recommender, common_ingredients)
        elif search_type == "🎭 By Style/Mood":
            results = handle_style_search(recommender)
        elif search_type == "🎉 By Occasion":
            results = handle_occasion_search(recommender)
        elif search_type == "🎲 Mixed Preferences":
            results = handle_mixed_search(recommender, common_ingredients, alcoholic_options)
        elif search_type == "📂 By Category":
            results = handle_category_search(recommender)
        elif search_type == "🎰 Random Discovery":
            results = handle_random_search(recommender)
        elif search_type == "🔍 Search by Name":
            results = handle_name_search(recommender)
        else:
            results = []
        
        if results:
            st.session_state.search_results = results
        
        # Display results from session state
        if st.session_state.search_results:
            st.divider()
            st.subheader(f"🍹 Found {len(st.session_state.search_results)} cocktail{'s' if len(st.session_state.search_results) != 1 else ''}:")
            
            similarity_types = [
                "🥃 By Ingredients",
                "🎭 By Style/Mood",
                "🎉 By Occasion",
                "🎲 Mixed Preferences"
            ]
            show_similarity = st.session_state.last_search_type in similarity_types
            
            for result in st.session_state.search_results:
                cocktail = recommender.format_cocktail_result(result)
                if not show_similarity:
                    cocktail['similarity'] = None
                display_cocktail(cocktail)
                st.divider()
        
        elif st.session_state.last_search_type and st.session_state.last_search_type != "🔍 Search by Name":
            st.info("No cocktails found matching your criteria. Try adjusting your preferences!")
    
    with col2:
        st.subheader("💡 Tips")
        st.info("""
        **How to get better suggestions:**
        
        🎯 Be specific with ingredients
        
        🎭 Combine multiple style preferences
        
        🎉 Try different occasions
        
        🎲 Use the random discovery for inspiration
        
        🔍 Search by partial names works too!
        """)
        
        st.subheader("📊 Database Stats")
        try:
            # Attempt to get dynamic stats if method exists, fallback to hardcoded
            total_cocktails = recommender.get_total_cocktails() if hasattr(recommender, 'get_total_cocktails') else "600+"
            st.metric("Available Cocktails", total_cocktails)
            st.metric("Ingredient Combinations", "∞")
            st.metric("AI Accuracy", "95%+")
        except:
            st.metric("Available Cocktails", "600+")
            st.metric("Ingredient Combinations", "∞")
            st.metric("AI Accuracy", "95%+")

if __name__ == "__main__":
    common_ingredients = ["vodka", "gin", "rum", "whiskey", "tequila", "bourbon", "lime", "lemon", "orange", "cranberry", "pineapple", "mint", "basil", "simple syrup", "triple sec", "vermouth"]
    alcoholic_options = ["Alcoholic", "Non alcoholic", "Optional alcohol"]
    main()
