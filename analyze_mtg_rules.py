import pandas as pd

def load_rules_questions(filename='data/mtg_rules_questions.csv'):
    return pd.read_csv(filename, parse_dates=['created_utc'])

def print_sample_questions(df, n=5):
    print(f"\nSample of {n} questions with full body text:")
    for i, (title, body) in enumerate(zip(df['title'].head(n), df['body'].head(n)), 1):
        print(f"\n{i}. Title: {title}")
        print(f"Body:\n{body}\n")
        print("-" * 80)  # Separator between questions

def print_basic_stats(df):
    print(f"\nTotal number of questions: {len(df)}")
    print(f"Date range: from {df['created_utc'].min()} to {df['created_utc'].max()}")
    print(f"Average score: {df['score'].mean():.2f}")
    print(f"Average number of comments: {df['num_comments'].mean():.2f}")

def print_question_length_stats(df):
    df['body_length'] = df['body'].str.len()
    print(f"\nQuestion body length statistics:")
    print(f"Average length: {df['body_length'].mean():.2f} characters")
    print(f"Median length: {df['body_length'].median()} characters")
    print(f"Shortest question: {df['body_length'].min()} characters")
    print(f"Longest question: {df['body_length'].max()} characters")

if __name__ == "__main__":
    mtg_rules = load_rules_questions()
    print_sample_questions(mtg_rules)
    print_basic_stats(mtg_rules)
    print_question_length_stats(mtg_rules)