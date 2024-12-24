import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def extract_date(filename):
    """
    Extracts the date from the filename.
    Expected format: YYYY-MM-DD_*.txt
    """
    match = re.match(r'(\d{4}-\d{2}-\d{2})_.*\.txt$', filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern.")

def count_words(filepath):
    """
    Counts the number of words in the given text file.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
        words = text.split()
        return len(words)

def load_word_counts(folder_path):
    """
    Loads all text files from the folder, extracts dates, counts words,
    and returns a DataFrame with dates and word counts.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder_path} is not a valid directory.")

    word_counts = {}

    for file in folder.glob('*.txt'):
        try:
            date = extract_date(file.name)
            count = count_words(file)
            # If multiple files have the same date, accumulate the counts
            if date in word_counts:
                word_counts[date] += count
            else:
                word_counts[date] = count
        except ValueError as e:
            print(f"Skipping file: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(list(word_counts.items()), columns=['Date', 'WordCount'])
    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    # Sort by date
    df = df.sort_values('Date')
    # Set 'Date' as the index
    df.set_index('Date', inplace=True)
    # Perform rolling median filter of window 5:
    df = df.rolling(window=3, center=True).median().rolling(window=10, center=True).mean()

    return df

def plot_word_counts(df):
    """
    Plots the word counts over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['WordCount'], marker='o', linestyle='-')
    plt.title('Word Count Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Words')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Specify the folder containing the text files
    folder_path = '/Users/vigji/Desktop/sando-data/whispered_backup'  # <-- Replace with your folder path

    # Load word counts
    df = load_word_counts(folder_path)
    print(df)

    # Plot the results
    plot_word_counts(df)

if __name__ == "__main__":
    main()
