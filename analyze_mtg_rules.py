import pandas as pd
import argparse
import textwrap

def read_and_print_posts(file_path, start_index, batch_size=5):
    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=['created_utc'])
    
    # Ensure start_index is within bounds
    start_index = max(0, min(start_index, len(df) - 1))
    
    # Get the batch of posts
    end_index = min(start_index + batch_size, len(df))
    batch = df.iloc[start_index:end_index]
    
    # Print the posts
    for i, (_, post) in enumerate(batch.iterrows(), start=start_index):
        print(f"\n--- Post {i + 1} ---")
        print(f"ID: {post['id']}")
        print("Body:")
        # Wrap the body text for better readability
        wrapped_body = textwrap.wrap(post['body'], width=80)
        for line in wrapped_body:
            print(line)
        print("-" * 80)
    
    print(f"\nDisplayed posts {start_index + 1} to {end_index} out of {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and display MTG rules questions from CSV.")
    parser.add_argument("start_index", type=int, nargs="?", default=0, 
                        help="Starting index for displaying posts (default: 0)")
    parser.add_argument("--file", type=str, default="data/mtg_rules_questions.csv",
                        help="Path to the CSV file (default: data/mtg_rules_questions.csv)")
    args = parser.parse_args()

    read_and_print_posts(args.file, args.start_index)