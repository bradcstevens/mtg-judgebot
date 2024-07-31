import os
from dotenv import load_dotenv
import praw
import pandas as pd
from datetime import datetime
import time

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

def get_last_post_id(filename='data/mtg_rules_questions.csv'):
    try:
        df = pd.read_csv(filename)
        if not df.empty and 'id' in df.columns:
            return df['id'].iloc[-1]
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
        pass
    return None

def fetch_mtg_rules_questions(reddit, last_post_id=None, total_limit=5000, batch_size=100):
    subreddit = reddit.subreddit('mtgrules')
    posts = []
    total_fetched = 0

    print(f"Fetching posts from r/{subreddit.display_name}")
    print(f"Starting after post ID: {last_post_id}")
    
    while total_fetched < total_limit:
        try:
            if last_post_id:
                batch = list(subreddit.new(limit=batch_size, params={'after': f't3_{last_post_id}'}))
            else:
                batch = list(subreddit.new(limit=batch_size))

            if not batch:
                print("No more posts to fetch.")
                break

            for post in batch:
                posts.append({
                    'id': post.id,
                    'title': post.title,
                    'body': post.selftext,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'url': post.url,
                    'flair': post.link_flair_text if post.link_flair_text else 'No Flair'
                })

            total_fetched += len(batch)
            last_post_id = batch[-1].id
            print(f"Fetched {total_fetched} posts so far... Last post ID: {last_post_id}")

            if total_fetched >= total_limit:
                break

            time.sleep(2)  # To avoid hitting rate limits

        except Exception as e:
            print(f"An error occurred while fetching posts: {e}")
            time.sleep(60)  # Wait for a minute before retrying

    return pd.DataFrame(posts)

def save_to_csv(df, filename='data/mtg_rules_questions.csv'):
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename, parse_dates=['created_utc'])
        print(f"Existing CSV has {len(existing_df)} posts")
        combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['id']).reset_index(drop=True)
        new_posts = len(combined_df) - len(existing_df)
        print(f"Added {new_posts} new posts")
    else:
        combined_df = df
        print(f"Created new CSV with {len(df)} posts")

    combined_df.to_csv(filename, index=False)
    print(f"Total posts in CSV: {len(combined_df)}")
    return combined_df

if __name__ == "__main__":
    try:
        reddit = get_reddit_instance()
        last_post_id = get_last_post_id()
        mtg_rules = fetch_mtg_rules_questions(reddit, last_post_id)
        full_df = save_to_csv(mtg_rules)
        
        print("\nSample of fetched posts:")
        print(full_df[['id', 'title', 'flair']].head())
        
        print(f"\nUnique flairs found: {full_df['flair'].unique()}")
        
        print("\nPost distribution by flair:")
        print(full_df['flair'].value_counts())
        
        print(f"\nTotal unique posts: {full_df['id'].nunique()}")
        print(f"Date range: from {full_df['created_utc'].min()} to {full_df['created_utc'].max()}")
        
    except Exception as e:
        print(f"An error occurred: {e}")