import wikipedia

def check_fact(statement):
    try:
        # Extract keywords (first 6 words are usually enough)
        query = " ".join(statement.split()[:6])
        summary = wikipedia.summary(query, sentences=2)

        # Check if statement terms are in summary
        if any(word.lower() in summary.lower() for word in statement.split()):
            return f"✅ Wikipedia says: {summary}"
        else:
            return f"⚠️ No direct match found.\n🔎 Wikipedia says: {summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"⚠️ Your query is too broad. Try one of: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return "❌ Fact not found on Wikipedia."
    except Exception as e:
        return f"❌ Error: {str(e)}"
