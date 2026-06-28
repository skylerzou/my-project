import requests, json, os
from dotenv import load_dotenv

load_dotenv()

# 1. Call API
response = requests.get(
    "https://www.pokemonpricetracker.com/api/v2/cards",
    headers={"Authorization": f"Bearer {os.getenv('PPT_API_KEY')}"},
    params={"search": "Charizard", "limit": 10}
)

# 2. Parse
data = response.json()

# 3. Save immediately
with open("response.json", "w") as f:
    json.dump(data, f, indent=2)

# 4. Pretty print
print(json.dumps(data, indent=2))
