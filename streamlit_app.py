import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4527A0;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5E35B1;
        margin-bottom: 2rem;
        text-align: center;
    }
    .prediction-result {
        padding: 20px;
        border-radius: 10px;
        background-color: #E8EAF6;
        margin: 20px 0;
        text-align: center;
        font-size: 2rem;
    }
    .stButton>button {
        background-color: #5E35B1;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #4527A0;
    }
    .form-section {
        background-color: #F5F7FF;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stExpander {
        border: 1px solid #E0E0E0;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description with custom styling
st.markdown('<h1 class="main-header">💻 Laptop Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict laptop prices based on specifications using machine learning</p>', unsafe_allow_html=True)

# Load the model and dataset directly in the Streamlit app
@st.cache(allow_output_mutation=True)
def load_model_and_data():
    current_dir = Path(__file__).parent
    pipe = pickle.load(open(current_dir / "pipe.pkl", "rb"))
    df = pickle.load(open(current_dir / "df.pkl", "rb"))
    return pipe, df

try:
    pipe, df = load_model_and_data()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.error("Please make sure the model files (pipe.pkl and df.pkl) are in the same directory as this app.")
    st.stop()

# Create a container for the form
with st.container():
    
    # Create two columns for the form
    col1, col2 = st.columns(2)

    # Form inputs
    with col1:
        st.subheader("📋 Basic Specifications")
        company = st.selectbox("Brand", ["Apple", "HP", "Dell", "Lenovo", "Asus", "Acer", "MSI", "Toshiba", "Samsung"])
        type_name = st.selectbox("Type", ["Ultrabook", "Notebook", "Gaming", "2 in 1 Convertible", "Workstation"])
        inches = st.slider("Screen Size (inches)", 10.0, 18.0, 15.0, 0.1)
        screen_resolution = st.selectbox("Screen Resolution", ["1920x1080", "1366x768", "3840x2160", "2560x1440", "2880x1800"])
        cpu = st.selectbox("CPU", ["Intel Core i7", "Intel Core i5", "AMD Ryzen 7", "AMD Ryzen 5", "Intel Core i9"])
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64])

    with col2:
        st.subheader("🔧 Additional Features")
        memory = st.selectbox("Storage", ["256GB SSD", "512GB SSD", "1TB HDD", "1TB SSD + 1TB HDD", "2TB SSD"])
        gpu = st.selectbox("GPU", ["NVIDIA GeForce GTX", "Intel Integrated", "AMD Radeon", "NVIDIA GeForce RTX"])
        opsys = st.selectbox("Operating System", ["Windows", "macOS", "Linux", "Chrome OS"])
        weight = st.slider("Weight (kg)", 0.5, 5.0, 2.0, 0.1)
        touchscreen = st.selectbox("Touchscreen", ["Yes", "No"])
        ips = st.selectbox("IPS Panel", ["Yes", "No"])
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction function
def predict_price(input_data):
    # Process the input data to match the model's expected format
    # Extract information from memory selection
    if "SSD" in input_data["memory"] and "HDD" in input_data["memory"]:
        # For combined storage like "1TB SSD + 1TB HDD"
        ssd_size = int(input_data["memory"].split("TB SSD")[0].split(" ")[-1]) * 1024
        hdd_size = int(input_data["memory"].split("TB HDD")[0].split("+ ")[-1]) * 1024
    elif "SSD" in input_data["memory"]:
        # For SSD only storage
        ssd_size = int(input_data["memory"].split("GB SSD")[0]) if "GB" in input_data["memory"] else int(input_data["memory"].split("TB SSD")[0]) * 1024
        hdd_size = 0
    elif "HDD" in input_data["memory"]:
        # For HDD only storage
        hdd_size = int(input_data["memory"].split("GB HDD")[0]) if "GB" in input_data["memory"] else int(input_data["memory"].split("TB HDD")[0]) * 1024
        ssd_size = 0
    else:
        ssd_size = 0
        hdd_size = 0
    
    # Extract GPU brand - ensure it matches expected categories
    if "NVIDIA" in input_data["gpu"]:
        gpu_brand = "Nvidia"
    elif "AMD" in input_data["gpu"]:
        gpu_brand = "AMD"
    elif "Intel" in input_data["gpu"]:
        gpu_brand = "Intel"
    else:
        gpu_brand = "Other"
    
    # Extract CPU brand - ensure it matches expected categories
    # Use the exact categories from the model's transformer
    if "Intel" in input_data["cpu"]:
        if "i7" in input_data["cpu"]:
            cpu_brand = "Intel Core i7"
        elif "i5" in input_data["cpu"]:
            cpu_brand = "Intel Core i5"
        elif "i3" in input_data["cpu"]:
            cpu_brand = "Intel Core i3"
        else:
            # For other Intel CPUs like i9 that weren't in the training data
            cpu_brand = "Other Intel Processor"
    elif "AMD" in input_data["cpu"]:
        cpu_brand = "AMD Processor"
    else:
        # Default to a known category if unknown
        cpu_brand = "Other Intel Processor"
    
    # Calculate PPI (pixels per inch) from screen resolution
    if "x" in input_data["screenResolution"]:
        width, height = map(int, input_data["screenResolution"].split("x"))
        diagonal_pixels = np.sqrt(width**2 + height**2)
        ppi = diagonal_pixels / float(input_data["inches"])
    else:
        ppi = 0
    
    # Create DataFrame with all required columns
    X = pd.DataFrame({
        'Company': [input_data["company"]],
        'TypeName': [input_data["typeName"]],
        'Inches': [float(input_data["inches"])],
        'Ram': [int(input_data["ram"])],
        'Weight': [float(input_data["weight"])],
        'Touchscreen': [1 if input_data["touchscreen"] == "Yes" else 0],
        'Ips': [1 if input_data["ips"] == "Yes" else 0],
        'ppi': [ppi],
        'Cpu brand': [cpu_brand],
        'HDD': [hdd_size],
        'SSD': [ssd_size],
        'Gpu brand': [gpu_brand],
        'os': [input_data["opsys"]]
    })
    
    # Make prediction and apply exponential transformation like in the Flask app
    predicted_price = np.exp(pipe.predict(X)[0])
    
    # Convert from rupees to USD (approximate conversion rate)
    usd_price = predicted_price / 83.0  # Current approximate INR to USD conversion rate
    
    # Return the result
    return {
        "predicted_price": int(usd_price)
    }

# Center the prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Prediction button
    predict_button = st.button("Predict Price 🚀", key="predict_button")

# Results section
if predict_button:
    # Show a spinner while calculating
    with st.spinner("Calculating price..."):
        # Prepare data for prediction
        input_data = {
            "company": company,
            "typeName": type_name,
            "inches": inches,
            "screenResolution": screen_resolution,
            "cpu": cpu,
            "ram": ram,
            "memory": memory,
            "gpu": gpu,
            "opsys": opsys,
            "weight": weight,
            "touchscreen": touchscreen,
            "ips": ips
        }
        
        # Make prediction
        try:
            result = predict_price(input_data)
            price = result['predicted_price']
            
            # Display prediction with animation
            st.balloons()
            st.markdown(f'<div class="prediction-result">Predicted Price: <span style="color:#4527A0; font-weight:bold">${price:,.2f}</span></div>', unsafe_allow_html=True)
            
            # Display a summary of selected specifications
            st.subheader("📊 Specification Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Brand:** {company}")
                st.markdown(f"**Type:** {type_name}")
                st.markdown(f"**CPU:** {cpu}")
                st.markdown(f"**RAM:** {ram} GB")
                st.markdown(f"**Storage:** {memory}")
                
            with col2:
                st.markdown(f"**GPU:** {gpu}")
                st.markdown(f"**Screen:** {inches}\" ({screen_resolution})")
                st.markdown(f"**OS:** {opsys}")
                st.markdown(f"**Weight:** {weight} kg")
                st.markdown(f"**Touchscreen:** {touchscreen}")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Add some information about the model
with st.expander("ℹ️ About the Model"):
    st.markdown("""
    This laptop price prediction model is trained on a dataset of laptop specifications and prices.
    It uses XGBoost regression to predict prices based on features like brand, CPU, RAM, and more.
    
    The model has been trained on data from major laptop manufacturers and can predict prices
    with reasonable accuracy for most consumer laptops.
    
    **Key features that influence laptop prices:**
    - CPU performance and brand
    - RAM capacity
    - Storage type (SSD vs HDD) and capacity
    - GPU performance
    - Screen size and resolution
    - Brand premium
    """)

# Add footer
st.markdown("""
---
<p style="text-align: center; color: #666666; font-size: 0.8rem;">
© 2023 Laptop Price Predictor | Built with Streamlit and Machine Learning
</p>
""", unsafe_allow_html=True)