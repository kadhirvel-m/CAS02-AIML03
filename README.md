## Team ID: CAS02

## Problem Statement ID: AIML03

## Problem Statement:

#### Sentiment analysis model to categorize tweets into positive, negative, or neutral sentiments.

## Approach

This system enhances the existing AI-Powered Social Media Sentiment Analysis for Brands by integrating Mistral AI, a large language model (LLM), to provide positive factors, negative factors, neutral factors and a business intelligence chatbot. The system will analyze tweets, classify sentiment, and use Mistral AI to generate actionable insights and recommendations for brands. The chatbot will allow users to interactively query the system for deeper insights.

## Tech Stack

#### Tech Stack

- Sentiment Analysis: RoBERTa (Hugging Face Transformers),VADER for sentiment classification.

#### Large Language Model (LLM):

- Mistral AI for generating insights and powering the chatbot.

#### Data Collection:

- Tweepy (Twitter API) for fetching tweets.

#### Data Preprocessing:

- spaCy and Regex for cleaning and preprocessing tweets.

#### Visualization:

- Matplotlib, Chartjs and Streamlit for interactive dashboards.

#### Backend:

- Flask for serving the Mistral AI model and chatbot.

## Uniqueness

#### Comprehensive Insights:

- Combines sentiment analysis with Mistral AI to provide positive factors, negative factors, neutral factors and recommendations.

##### Interactive Chatbot:

A business intelligence chatbot powered by Mistral AI allows users to query the system for deeper insights (e.g., "What are the most common complaints about my brand?").

##### Real-Time Analysis:

Fetches and analyzes tweets in real-time, providing up-to-date insights.

##### Actionable Recommendations:

Goes beyond sentiment classification to offer actionable business intelligence.

## Feasibility

#### Technical Feasibility

- All components (RoBERTa, VADER, Mistral AI, Tweepy, etc.) are well-documented and widely used.
- Mistral AI can be integrated via APIs or locally hosted models.
- The system can be built using open-source tools, reducing costs.

#### Economic Feasibility

- The system provides high value to brands by offering actionable insights, making it a worthwhile investment.

#### Operational Feasibility

- The system is user-friendly, with an interactive drag and drop dashboard and chatbot.
- It can be easily maintained and updated with new models or features.

## Deliverables

At the end of the **AI-Powered Social Media Sentiment Analysis with Mistral AI Integration** project, the following will be delivered:

#### 1. Sentiment Analysis Model

- Fine-tuned **RoBERTa model** and **VADER model** for classifying tweets into **positive**, **negative**, or **neutral** sentiments.
- Preprocessing scripts to clean and prepare social media data (e.g., removing URLs, special characters, and stopwords).

#### 2. Mistral AI Integration

- Module to generate:
  - **Positive Factors**: What customers like about the brand.
  - **Negative Factors**: Common complaints or issues.
  - **Neutral Factors**: Neutral comments about the brand
  - **Actionable Recommendations**: Steps to address negative feedback.
- **Business Intelligence Chatbot** powered by Mistral AI for interactive querying and insights.

#### 3. Interactive Dashboard

- Flask based dashboard with:
  - Sentiment distribution (e.g., pie charts, bar graphs).
  - Positive, negative and neutral factors.
  - Recommendations for improvement.
  - Real-time sentiment trends over time.
- Interactive visualizations using **ChartJS** or **Matplotlib**.

#### 4. Real-Time Tweet Fetcher

- **Tweepy**-based module to fetch tweets in real-time using the **Twitter API**.
- Supports user-defined keywords, hashtags, or brand names.

#### 5. Business Intelligence Chatbot

- Chatbot interface for:
  - Asking questions (e.g., "What are the most common complaints?").
  - Getting insights (e.g., "Show me sentiment trends for the last month").

#### 6. Exportable Reports

- Functionality to export sentiment analysis results and insights in **CSV** format.
