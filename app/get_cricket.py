import requests
import dateparser

# 🧠 Get team codes and date
def extract_match_details(text):
    teams = ["CSK", "KKR", "RCB", "MI", "GT", "LSG", "SRH", "RR", "DC", "PBKS"]
    mentioned = [team for team in teams if team in text.upper()]
    parsed_date = dateparser.parse(text)
    match_date = parsed_date.date() if parsed_date else None
    return mentioned, match_date

# 🏏 CricAPI Key (Replace with your actual key or store in .env later)
CRIC_API_KEY = "acd0b40c-b742-408d-ad12-94024ad43912"

# 🔍 Match validation
def validate_cricket_fact(user_input):
    teams, _ = extract_match_details(user_input)
    if len(teams) < 2:
        return "⚠️ Could not detect 2 valid teams."

    # Use CricAPI's recent matches endpoint
    match_url = f"https://api.cricapi.com/v1/recentMatches?apikey={CRIC_API_KEY}&offset=0"

    try:
        res = requests.get(match_url)
        data = res.json()
    except Exception as e:
        return f"❌ Failed to fetch data: {e}"

    if data.get("status") != "success":
        return f"❌ CricAPI Error: {data.get('message', 'Unknown error')}"

    for match in data["data"]:
        team1 = match["teams"][0].upper()
        team2 = match["teams"][1].upper()

        if set(teams).issubset({team1, team2}):
            result = match.get("status", "").strip()
            if not result or "won" not in result.lower():
                return f"⚠️ Match found but no winner recorded yet.\n📋 Status: {result}"

            if teams[0] in result.upper():
                return f"✅ Fact Check: Yes, {teams[0]} won."
            elif teams[1] in result.upper():
                return f"❌ Fact Check: Actually, {teams[1]} won — not {teams[0]}."
            else:
                return f"⚠️ Unable to determine winner.\n📋 Status: {result}"

    return "⚠️ No recent match found between those teams."
