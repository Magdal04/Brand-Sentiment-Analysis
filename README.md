# Brand Analysis: AI-Driven Brand Sentiment Analysis

![Python](https://img.shields.io/badge/Python-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_AI_Studio-8E75B2?logo=google&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-VADER_%7C_LDA-orange)

An end-to-end data pipeline and reactive dashboard that automatically analyzes public perception of major brands (e.g., Nike, Toyota, McDonald's) using Natural Language Processing (NLP) and Generative AI. 

Rather than just showing simple positive/negative charts, this tool understands *why* people feel a certain way and acts as an **on-demand AI business consultant** to generate actionable, department-specific strategies.

## Key Features
* **Executive Overview:** Real-time sentiment distribution (Positive/Neutral/Negative) and AI-generated text summaries of the current brand situation.
* **Topic Intelligence:** Uses Latent Dirichlet Allocation (LDA) and TF-IDF to extract underlying topics and keywords driving consumer conversation.
* **AI Consultant (Gemini):** Instantly calculates Department Health Scores and generates actionable, persona-driven business recommendations (Marketing, Operations, Product) using Google's Gemini API.
* **Evidence Tracking:** Directly connects sentiment scores to the raw underlying comments and news articles.
* **Historical Trends:** Tracks brand sentiment over time to measure campaign success or crisis recovery.

## Architecture
1. **Data Orchestration:** Managed via automated workflows (e.g., n8n) triggering web scrapers.
2. **Data Processing (NLP):** Python pipeline handles cleaning (emoji parsing, stop words), sentiment scoring (VADER/Transformers), and Topic Modeling.
3. **Backend Engine (FastAPI):** Serves the processed JSON payload to the frontend dynamically.
4. **Interactive Dashboard (Streamlit):** Visualizes the data and makes direct asynchronous calls to the Gemini API for strategic insights.

## Author
**Magdal04**
* GitHub: [https://github.com/Magdal04](https://github.com/Magdal04)

---
*Built as a capstone/portfolio project showcasing Data Engineering, NLP, and Full-Stack GenAI integration.*


