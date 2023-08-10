import os
import tweepy

# Read the credentials from environment variables
API_KEY = os.environ.get("TWITTER_API_KEY")
API_SECRET_KEY = os.environ.get("TWITTER_API_SECRET_KEY")
ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

# Set up the tweepy authorization
auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# Create the tweepy API object
api = tweepy.API(auth)

# Fetch user information by username
user = api.get_user(screen_name="USERNAME")

print(user.name)  # User's name
print(user.profile_image_url)  # URL of the profile image
