import os
import tweepy
from dotenv import load_dotenv


# Load the .env file
load_dotenv()

# Read the credentials from environment variables
API_KEY = os.environ.get("TWITTER_API_KEY")
API_SECRET_KEY = os.environ.get("TWITTER_API_SECRET_KEY")
ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

if not all([API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_TOKEN_SECRET]):
    raise ValueError("One or more environment variables are not set.")


# Set up the tweepy authorization
auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# Create the tweepy API object
api = tweepy.API(auth)

# Fetch user information by username
def get_user_data(screen_name):
    cache_file = f"cache_{screen_name}.json"
    
    # If cache exists, load it
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Otherwise, fetch data from API
    user = api.get_user(screen_name=screen_name)
    data = {
        "name": user.name,
        "profile_image_url": user.profile_image_url,
        "tweets": []
    }
    
    # Fetch the first 100 tweets
    tweets = api.user_timeline(screen_name=screen_name, count=100)
    for tweet in tweets:
        data["tweets"].append(tweet.text)
    
    # Cache the data
    with open(cache_file, 'w') as f:
        json.dump(data, f)
    
    return data

# Test
user_data = get_user_data("lantos1618")
print(user_data["name"])
print(user_data["profile_image_url"])
print(user_data["tweets"][:5])  # Print the first 5 tweets