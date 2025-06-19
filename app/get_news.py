import requests
import dateparser
from datetime import date
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("NEWS_API_KEY")

def fetch_news(query):
    if not api_key:
        print("❌ API Key not loaded.")
        return ["⚠️ API key missing."]

    # Try to extract a date
    parsed_date = dateparser.parse(query)
    use_date_filter = bool(parsed_date)
    search_date = parsed_date.date() if parsed_date else date.today()

    # Remove date-related words from the query (optional improvement)
    keywords_only = " ".join([word for word in query.split() if not word.isdigit()])

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keywords_only,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 10,
        "apiKey": api_key
    }

    if use_date_filter:
        params["from"] = search_date
        params["to"] = search_date

    print(f"📅 Query: {keywords_only} | Date Filter: {use_date_filter} ({search_date})")

    response = requests.get(url, params=params)
    data = response.json()

    print("🔁 NewsAPI Raw Response:", data)

    if data.get("status") == "ok":
        articles = data.get("articles", [])
        if not articles:
            return ["⚠️ No articles found for this topic."]
        return [article["title"] for article in articles]
    else:
        return [f"⚠️ Error: {data.get('message', 'Unknown error')}"]
