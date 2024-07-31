import os
from dotenv import load_dotenv
import praw
import pandas as pd
from datetime import datetime

load_dotenv('.env.local')

def get_reddit_instance():
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_SECRET')
    reddit_username = os.getenv('REDDIT_USERNAME')

    print("\nEnvironment variables:")
    print(f"REDDIT_CLIENT_ID: {'Set' if client_id else 'Not set'}")
    print(f"REDDIT_SECRET: {'Set' if client_secret else 'Not set'}")
    print(f"REDDIT_USERNAME: {'Set' if reddit_username else 'Not set'}")

    if not all([client_id, client_secret, reddit_username]):
        raise ValueError("Please set REDDIT_CLIENT_ID, REDDIT_SECRET, and REDDIT_USERNAME in .env.local file.")

    return praw.Reddit(client_id=client_id,
                       client_secret=client_secret,
                       user_agent=f'python:mtg.rules.fetcher:v1.0 (by /u/{reddit_username})')

def fetch_mtg_rules_questions(reddit, limit=100):
    subreddit = reddit.subreddit('mtgrules')
    posts = []

    print(f"Fetching posts from r/{subreddit.display_name}")
    
    for post in subreddit.new(limit=limit):
        posts.append({
            'title': post.title,
            'body': post.selftext,
            'created_utc': datetime.fromtimestamp(post.created_utc),
            'score': post.score,
            'num_comments': post.num_comments,
            'url': post.url,
            'flair': post.link_flair_text
        })

    return pd.DataFrame(posts)

def save_to_csv(df, filename='mtg_rules_questions.csv'):
    df.to_csv(filename, index=False)
    print(f"Fetched {len(df)} posts and saved to {filename}")

if __name__ == "__main__":
    try:
        reddit = get_reddit_instance()
        mtg_rules = fetch_mtg_rules_questions(reddit)
        save_to_csv(mtg_rules)
        
        print("\nSample of fetched posts:")
        print(mtg_rules[['title', 'flair']].head())
        
        print(f"\nUnique flairs found: {mtg_rules['flair'].unique()}")
        
        print("\nPost distribution by flair:")
        print(mtg_rules['flair'].value_counts())
        
    except Exception as e:
        print(f"An error occurred: {e}")