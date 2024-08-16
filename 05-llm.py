from transformers import pipeline

# Initialize the text generation pipeline
text_generator = pipeline("text-generation")

# Prompt the LLM with a starting sentence
prompt = "Once upon a time, there was a beautiful cat..."

# Generate text based on the prompt
generated_text = text_generator(prompt)

# Print the generated text
print(generated_text[0]["generated_text"])

##################

from transformers import pipeline
import yfinance as yf
from bs4 import BeautifulSoup
import requests

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1y')
    return data

def analyze_sentiment(text):
    sentiment_analyzer = pipeline('sentiment-analysis')
    result = sentiment_analyzer(text)
    return result[0]['label']



def get_stock_news(ticker):
    #url = f'https://finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch'
    url = f'https://finance.yahoo.com/quote/AAPL/?p=AAPL&.tsrc=fin-srch'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [headline.text for headline in soup.find_all('h3', class_='Mb(5px)')]
    return response

#Bringing It All Together:
#Now, let’s combine these components to create a holistic analysis:

# Example usage for a stock (e.g., Apple - AAPL)
stock_ticker = 'AAPL'

# Get stock data
stock_data = get_stock_data(stock_ticker)

# Get stock news
news_headlines = get_stock_news(stock_ticker)

# Analyze sentiment for each news headline
sentiments = [analyze_sentiment(headline) for headline in news_headlines]

# Print stock data
print(f"Stock Data for {stock_ticker}:\n{stock_data}")

# Print news headlines and sentiments
print("\nNews Headlines:")
for headline, sentiment in zip(news_headlines, sentiments):
    print(f"- {headline} (Sentiment: {sentiment})")