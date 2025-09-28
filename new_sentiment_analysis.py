import pandas as pd
import matplotlib.pyplot as plt
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon (only needed once)
nltk.download("vader_lexicon")

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Load queries from CSV
data = pd.read_csv("data/queries.csv")

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to calculate sentiment
def get_sentiment(text):
    if not isinstance(text, str) or not text.strip():  # Handle empty/NaN
        return "Neutral"
    score = sia.polarity_scores(text)["compound"]  # -1 (neg) to +1 (pos)
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
data["Sentiment"] = data["query"].apply(get_sentiment)

# Save results
data.to_csv("output/sentiment_results.csv", index=False)
print("Sentiment analysis completed. Results saved to output/sentiment_results.csv")

# Plot sentiment distribution
sentiment_counts = data["Sentiment"].value_counts()

# Map colors explicitly to avoid mismatch
color_map = {"Positive": "green", "Negative": "red", "Neutral": "gray"}
colors = [color_map[label] for label in sentiment_counts.index]

plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
plt.title("Sentiment Distribution (VADER)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Queries")
plt.savefig("output/sentiment_distribution.png")
plt.show()
