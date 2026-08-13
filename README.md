# 🍎 Apple Product Intelligence Center

An end-to-end AI-powered analytics platform for Apple product pricing intelligence, built using **Streamlit, XGBoost, LangChain, Google Gemini, and Qdrant Vector Database**.

The platform combines:

* 📊 Business Intelligence Dashboard
* 🤖 AI-Powered Market Assistant
* 🔍 RAG-Based Product Search
* 📈 Machine Learning Price Prediction
* 🗄️ Vector Database Search
* 📉 Product Lifecycle Analytics

---

# 🚀 Project Overview

Apple products experience continuous changes in pricing, discounts, ratings, inventory levels, and sale-event performance across e-commerce platforms.

This project provides a centralized intelligence platform that enables users to:

* Monitor Apple product pricing trends
* Compare Amazon and Flipkart pricing
* Analyze discount behavior
* Evaluate product ratings and reviews
* Forecast future product prices
* Ask questions using AI assistants
* Search historical product information using RAG

---

# ✨ Key Features

## 📊 Interactive Analytics Dashboard

Explore Apple product data through an interactive Streamlit dashboard.

### Available Analytics

### Pricing Analysis

* Average product prices
* Launch vs current price comparison
* Product category pricing insights

### Discount Analysis

* Deepest discounted products
* Discount trends over time
* Discount vs Rating correlation

### Platform Comparison

* Amazon vs Flipkart pricing
* Platform-wise discounts
* Platform-wise ratings

### Inventory Analytics

* Stock availability tracking
* Out-of-stock analysis
* Inventory trends

### Ratings & Reviews

* Category-wise ratings
* Most reviewed products
* Rating distributions

### Product Lifecycle Trends

* Price evolution over time
* Product depreciation analysis
* Product performance tracking

---

## 📈 Machine Learning Price Prediction

Predict future prices of Apple products using an XGBoost regression model.

### Prediction Inputs

* Product Category
* Product Model
* Platform
* Product Condition
* Stock Status
* Product Rating
* Review Count
* Launch Price
* Target Date

### Prediction Outputs

* Future Product Price
* Price Change Percentage
* Price Trend Visualization
* Long-Term Price Forecasting

### Machine Learning Stack

* XGBoost
* Scikit-Learn
* Joblib
* Pandas

---

## 🤖 AI Assistant

A Gemini-powered assistant that helps users understand dashboard insights.

Example Questions:

* Which Apple category has the highest average price?
* Which products receive the highest discounts?
* Compare MacBook and iPhone pricing.
* Explain recent pricing trends.

Powered by:

* Google Gemini
* Streamlit Chat Interface

---

## 🔍 RAG-Based Apple Pricing Assistant

Retrieval-Augmented Generation (RAG) chatbot built using:

* LangChain
* Qdrant Vector Database
* HuggingFace Embeddings
* Google Gemini

The chatbot retrieves relevant Apple product information from the vector database before generating responses.

### Example Questions

* What is the launch price of iPhone 15 Pro Max?
* Which Apple Watches have the highest ratings?
* Compare pricing of MacBook Air and MacBook Pro.
* Which products received the highest discounts?

---

# 🏗️ System Architecture

```text
User
 │
 ▼
Streamlit Dashboard
 │
 ├── Analytics Engine
 │
 ├── XGBoost Prediction Engine
 │
 ├── Gemini AI Assistant
 │
 └── RAG Assistant
         │
         ▼
    LangChain
         │
         ▼
  Qdrant Vector Database
         │
         ▼
Apple Product Dataset
```

---

# 🛠️ Tech Stack

## Frontend

* Streamlit

## Data Processing

* Pandas
* NumPy

## Visualization

* Plotly

## Machine Learning

* XGBoost
* Scikit-Learn
* Joblib

## Large Language Models

* Google Gemini

## RAG Components

* LangChain
* Qdrant
* HuggingFace Embeddings
* Sentence Transformers

## Deployment

* Streamlit Cloud

---

# 📂 Project Structure

```text
Apple-Product-Intelligence-Center/
│
├── main.py
├── ingest.py
├── rag_chain.py
│
├── requirements.txt
├── model_lookup.json
├── xgb_price_model_corrected.joblib
│
├── apple_products_pricing_2020_2026.csv
├── apple_icon.png
│
├── .env
├── .gitignore
│
└── README.md
```

---

# ⚙️ Installation

## Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Apple-Product-Intelligence-Center.git

cd Apple-Product-Intelligence-Center
```

---

## Step 2: Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root.

```env
QDRANT_URL=your_qdrant_url

QDRANT_API_KEY=your_qdrant_api_key

GOOGLE_API_KEY=your_google_gemini_api_key
```

---

# 🗄️ Create Vector Database

Run the ingestion script:

```bash
python ingest.py
```

This script:

* Loads Apple pricing data
* Generates embeddings
* Creates Qdrant collection
* Stores vector representations

---

# ▶️ Run Application

```bash
streamlit run main.py
```

After launching:

```text
http://localhost:8501
```

will open automatically.

---

# ☁️ Streamlit Cloud Deployment

## Required Secrets

Add these secrets in:

**Streamlit Cloud → App Settings → Secrets**

```toml
GEMINI_API_KEY="your_gemini_api_key"

QDRANT_URL="your_qdrant_url"

QDRANT_API_KEY="your_qdrant_api_key"
```

---

# 📈 Business Use Cases

This project can be used for:

### Retail Intelligence

Monitor product pricing across marketplaces.

### Pricing Strategy

Analyze discount and sale-event effectiveness.

### Market Research

Understand product lifecycle and pricing trends.

### Inventory Monitoring

Track stock availability and demand patterns.

### AI-Powered Product Insights

Ask natural language questions about product data.

---

# 📊 Dataset Information

The dataset contains Apple product information from 2020–2026 including:

* Product Name
* Product Category
* Launch Price
* Current Price
* Platform
* Ratings
* Reviews
* Stock Status
* Sale Events
* Product Condition

Categories Included:

* iPhone
* Mac
* iPad
* Apple Watch

---

# 🔮 Future Improvements

Planned enhancements:

* Multi-brand support
* Real-time pricing APIs
* Advanced forecasting models
* Product recommendation engine
* Customer sentiment analysis
* Automated report generation
* Dashboard export to PDF

---

# 👨‍💻 Author

## Chirayu Lokhande

B.Tech Computer Science (Big Data Analytics)

Data Analyst | Machine Learning Enthusiast | AI Developer

### Skills

* Python
* SQL
* Power BI
* Streamlit
* Machine Learning
* LangChain
* Vector Databases
* Generative AI
* Data Analytics

### Connect With Me
🚀 Live Demo
🔗 Streamlit App:
    https://apple-price-intelligence.streamlit.app

LinkedIn:
www.linkedin.com/in/chirayulokhande07

GitHub:
https://github.com/Chirayu0703

---

⭐ If you found this project useful, please consider giving it a star.

