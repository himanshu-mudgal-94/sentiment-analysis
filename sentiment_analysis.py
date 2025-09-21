from textblob import TextBlob

queries = [
  "What are your shipping options?",
  "How do I reset my password?",
  "Do you offer international delivery?",
  "Can I track my order?",
  "What is your return policy?",
  "How long does delivery take?",
  "Do you provide bulk discounts?",
  "Is there a warranty on this product?"
]

for q in queries:
    analysis = TextBlob(q)
    polarity = analysis.sentiment.polarity   # -1 (negative) to +1 (positive)

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print(f"{q} --> {sentiment}")
