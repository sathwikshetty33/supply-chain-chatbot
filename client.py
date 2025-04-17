import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Fresh Produce Analysis",
    page_icon="🍏",
    layout="wide"
)

# App title with styling
st.title("🥦 Fresh Produce Spoilage Analysis")
st.markdown("### Analyze cold chain and logistics data for fresh produce")

# Sidebar for model selection and data exploration
with st.sidebar:
    st.header("Settings")
    
    # Model selection
    model_options = ["Deepseek", "Llama", "Gemma"]
    selected_model = st.selectbox("Select AI Model", model_options)
    
    st.divider()
    
    # Data exploration options
    st.header("Data Explorer")
    
    # Function to fetch products from the API
    @st.cache_data
    def fetch_products():
        try:
            base_url = "http://localhost:8000"
            response = requests.get(f"{base_url}/products")
            if response.status_code == 200:
                return response.json().get("products", [])
            return []
        except:
            return []
    
    products = fetch_products()
    if products:
        selected_product = st.selectbox("Filter by Product", ["All Products"] + products)
        if st.button("Show Product Data"):
            st.session_state["show_product_data"] = True
            st.session_state["selected_product"] = selected_product
    
    st.divider()
    
    # Example questions
    st.header("Example Questions")
    example_questions = [
        "What produces have the highest spoilage rates?",
        "How does transit time affect spoilage?",
        "Compare spoilage rates between refrigerated and ambient storage",
        "Which products should be prioritized for delivery?",
        "What's the relationship between temperature and spoilage for berries?",
        "Which vegetables need cooler truck shipping?"
    ]
    
    for question in example_questions:
        if st.button(question, key=question):
            st.session_state["question"] = question

# Main content area divided into tabs
tab1, tab2 = st.tabs(["🔍 Query Analysis", "📊 Data Visualization"])

with tab1:
    # Input field for the question
    question = st.text_input(
        "Ask a question about the fresh produce data:",
        value=st.session_state.get("question", ""),
        key="question_input"
    )
    
    # Button to submit the question
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Analyze", type="primary")
    
    with col2:
        st.markdown("")  # Empty space for layout
    
    if submit_button:
        if not question.strip():
            st.error("Please enter a question.")
        else:
            # Show loading spinner
            with st.spinner(f"Analyzing with {selected_model} model..."):
                # Determine the endpoint based on the selected model
                base_url = "http://localhost:8000"  # Replace with your FastAPI server URL
                if selected_model == "Deepseek":
                    endpoint = f"{base_url}/deepseek"
                elif selected_model == "Llama":
                    endpoint = f"{base_url}/llama"
                elif selected_model == "Gemma":
                    endpoint = f"{base_url}/gemma"
                
                # Send the request to the FastAPI server
                payload = {"question": question}
                try:
                    response = requests.post(endpoint, json=payload, timeout=60)
                    response.raise_for_status()
                    answer = response.json().get("answer", "No answer received.")
                except requests.exceptions.RequestException as e:
                    answer = f"Error: {e}"
                
                # Display the answer in a nice card
                st.markdown("### Analysis Results")
                with st.container(border=True):
                    st.markdown(answer)
                
                # Option to try a different model
                st.markdown("---")
                st.markdown("##### Not satisfied? Try a different model or refine your question.")

with tab2:
    # Sample data for visualization
    # In a real app, you'd fetch this from your API or database
    sample_data = {
        "Product": ["Mangoes", "Cauliflower", "Cherries", "Blueberries", "Papaya", "Green Peas", 
                    "Peaches", "Plums", "Okra", "Tomatoes", "Mushrooms", "Lettuce", "Zucchini",
                    "Spinach", "Carrots", "Strawberries", "Cucumbers", "Pineapples"],
        "Category": ["Fruit", "Vegetable", "Fruit", "Fruit", "Fruit", "Vegetable", 
                     "Fruit", "Fruit", "Vegetable", "Vegetable", "Vegetable", "Vegetable", 
                     "Vegetable", "Vegetable", "Vegetable", "Fruit", "Vegetable", "Fruit"],
        "Spoilage (%)": [21.3, 29.2, 20.5, 20.7, 22.0, 34.9, 24.4, 34.6, 25.3, 30.6, 28.6, 
                         27.5, 19.0, 24.3, 20.5, 13.3, 25.9, 5.2],
        "Avg_Temperature": [16.6, 20.4, 15.7, 27.7, 27.3, 23.8, 17.1, 26.0, 29.2, 34.4, 
                           27.9, 25.7, 30.1, 34.5, 28.2, 16.5, 27.3, 19.9],
        "Transit_Time": [52, 71, 80, 106, 35, 50, 71, 64, 63, 105, 31, 111, 10, 73, 
                        34, 81, 60, 87],
        "Storage": ["Refrigerated", "Refrigerated", "Refrigerated", "Ambient", "Refrigerated",
                    "Ambient", "Ambient", "Refrigerated", "Refrigerated", "Ambient", 
                    "Ambient", "Refrigerated", "Ambient", "Ambient", "Refrigerated", 
                    "Refrigerated", "Refrigerated", "Refrigerated"]
    }
    df_viz = pd.DataFrame(sample_data)
    
    # Visualization options
    viz_options = [
        "Spoilage by Product",
        "Spoilage by Category",
        "Temperature vs Spoilage",
        "Transit Time vs Spoilage",
        "Storage Type Comparison"
    ]
    
    selected_viz = st.selectbox("Select Visualization", viz_options)
    
    # Create the selected visualization
    st.markdown("### " + selected_viz)
    
    fig, ax = plt.figure(figsize=(10, 6)), plt.subplot()
    
    if selected_viz == "Spoilage by Product":
        sorted_df = df_viz.sort_values("Spoilage (%)", ascending=False)
        sns.barplot(data=sorted_df, x="Product", y="Spoilage (%)", hue="Category", ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
    elif selected_viz == "Spoilage by Category":
        sns.boxplot(data=df_viz, x="Category", y="Spoilage (%)", ax=ax)
        
    elif selected_viz == "Temperature vs Spoilage":
        sns.scatterplot(data=df_viz, x="Avg_Temperature", y="Spoilage (%)", 
                        hue="Category", size="Transit_Time", sizes=(20, 200), ax=ax)
        plt.xlabel("Average Temperature (°C)")
        plt.ylabel("Spoilage (%)")
        
    elif selected_viz == "Transit Time vs Spoilage":
        sns.scatterplot(data=df_viz, x="Transit_Time", y="Spoilage (%)", 
                        hue="Category", style="Storage", ax=ax)
        plt.xlabel("Transit Time (Hours)")
        plt.ylabel("Spoilage (%)")
        
    elif selected_viz == "Storage Type Comparison":
        sns.boxplot(data=df_viz, x="Storage", y="Spoilage (%)", hue="Category", ax=ax)
    
    st.pyplot(plt.gcf())
    
    # Additional data explanation
    with st.expander("About the Visualizations"):
        st.markdown("""
        These visualizations help illustrate key relationships in the fresh produce dataset:
        
        - **Spoilage by Product**: Shows which products have the highest spoilage rates
        - **Spoilage by Category**: Compares spoilage rates between fruits and vegetables
        - **Temperature vs Spoilage**: Illustrates how temperature affects spoilage rates
        - **Transit Time vs Spoilage**: Shows the relationship between time in transit and spoilage
        - **Storage Type Comparison**: Compares spoilage rates between refrigerated and ambient storage
        
        The size of points in scatter plots represents additional variables like quantity or transit time.
        """)

# Display product data if requested from sidebar
if st.session_state.get("show_product_data", False):
    st.markdown("---")
    st.markdown(f"### Product Details: {st.session_state.get('selected_product', 'All Products')}")
    
    if st.session_state.get("selected_product") == "All Products":
        # Show a summary table for all products
        st.markdown("#### Summary by Product")
        product_summary = pd.DataFrame({
            "Product": sample_data["Product"],
            "Category": sample_data["Category"],
            "Avg Spoilage (%)": sample_data["Spoilage (%)"],
            "Avg Temperature (°C)": sample_data["Avg_Temperature"],
            "Avg Transit Time (hrs)": sample_data["Transit_Time"],
            "Storage Type": sample_data["Storage"]
        })
        st.dataframe(product_summary, use_container_width=True, hide_index=True)
    else:
        # Show details for the selected product
        selected = st.session_state.get("selected_product")
        product_data = [i for i, p in enumerate(sample_data["Product"]) if p == selected]
        
        if product_data:
            idx = product_data[0]
            st.markdown(f"**Category:** {sample_data['Category'][idx]}")
            st.markdown(f"**Avg Spoilage:** {sample_data['Spoilage (%)'][idx]}%")
            st.markdown(f"**Avg Temperature:** {sample_data['Avg_Temperature'][idx]}°C")
            st.markdown(f"**Avg Transit Time:** {sample_data['Transit_Time'][idx]} hours")
            st.markdown(f"**Typical Storage:** {sample_data['Storage'][idx]}")
            
            # Show recommended handling
            spoilage = sample_data['Spoilage (%)'][idx]
            if spoilage > 30:
                recommendation = "Critical attention needed. Use expedited shipping and controlled temperature."
            elif spoilage > 20:
                recommendation = "Use refrigerated transport and minimize transit time."
            else:
                recommendation = "Standard shipping procedures with normal monitoring."
                
            st.markdown(f"**Recommendation:** {recommendation}")
            
# Footer
st.markdown("---")
st.caption("Fresh Produce Analysis Tool - Data updated April 2025")