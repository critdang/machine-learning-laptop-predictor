# Laptop Price Predictor

A machine learning application that predicts laptop prices based on their specifications. This project uses a dataset of laptop specifications and their corresponding prices to train a model that can predict the price of a laptop given its features.

## Overview

This application allows users to input laptop specifications such as:
- Brand (Apple, HP, Dell, Lenovo, Asus, Acer, MSI, Toshiba, Samsung)
- Type (Ultrabook, Notebook, Gaming, 2 in 1 Convertible, Workstation)
- Processor type (Intel Core i7, Intel Core i5, AMD Ryzen 7, AMD Ryzen 5, Intel Core i9)
- RAM (4GB, 8GB, 16GB, 32GB, 64GB)
- Storage type and capacity (256GB SSD, 512GB SSD, 1TB HDD, 1TB SSD + 1TB HDD, 2TB SSD)
- Screen size (10-18 inches) and resolution (1920x1080, 1366x768, 3840x2160, etc.)
- GPU (NVIDIA GeForce GTX, Intel Integrated, AMD Radeon, NVIDIA GeForce RTX)
- Operating system (Windows, macOS, Linux, Chrome OS)
- Weight (0.5-5.0 kg)
- Touchscreen capability (Yes/No)
- IPS Panel (Yes/No)

The model then predicts the price based on these specifications.

## Data Source

The dataset used for this project is sourced from Kaggle's "Laptop Price Dataset" which contains information about various laptops and their prices. The dataset includes features like brand, processor details, RAM, storage, display specifications, and more.

## Model

The prediction model is built using:
- Scikit-learn for preprocessing and pipeline construction
- XGBoost for the regression model

The model is trained on historical laptop data and can predict prices for new laptop configurations.

## Project Structure

- `streamlit_app.py`: The main Streamlit application file
- `pipe.pkl`: Serialized machine learning pipeline
- `df.pkl`: Serialized DataFrame containing the processed dataset
- `requirements.txt`: List of Python dependencies

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd laptop-predictor
```
2. Setup Python Environment (recommended Python 3.9+):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```
4. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```
This will start the application and open it in your default web browser. If it doesn't open automatically, you can access it at http://localhost:8501 .

## Usage
1. Select the laptop specifications using the provided input widgets
2. The predicted price will be displayed based on your selections
3. Experiment with different configurations to see how they affect the price
## Future Improvements
- Add more recent laptop data to improve prediction accuracy
- Implement feature importance visualization
- Add comparison functionality between different laptop configurations
## License

MIT License - Feel free to use this project for any purpose, including commercial applications.

## Acknowledgements
- The dataset used in this project is from Kaggle
- Built with Streamlit, scikit-learn, and XGBoost