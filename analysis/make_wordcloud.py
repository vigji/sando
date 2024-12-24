import os
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords

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
    Also concatenates all text into a single string.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder_path} is not a valid directory.")

    word_counts = {}
    all_text = []

    for file in folder.glob('*.txt'):
        try:
            date = extract_date(file.name)
            count = count_words(file)
            # Read the text for concatenation
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                all_text.append(text)
            # If multiple files have the same date, accumulate the counts
            if date in word_counts:
                word_counts[date] += count
            else:
                word_counts[date] = count
        except ValueError as e:
            print(f"Skipping file: {e}")

    # Concatenate all text
    concatenated_text = '\n'.join(all_text)

    # Convert to DataFrame
    df = pd.DataFrame(list(word_counts.items()), columns=['Date', 'WordCount'])
    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    # Sort by date
    df = df.sort_values('Date')
    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    return df, concatenated_text

def save_concatenated_text(folder_path, concatenated_text):
    """
    Saves the concatenated text to 'all_concatenated.txt' in the specified folder.
    """
    output_file = Path(folder_path) / 'all_concatenated.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(concatenated_text)
    print(f"All text concatenated and saved to {output_file}")

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

def generate_wordclouds_per_year(folder_path, all_text):
    """
    Generates and saves a word cloud for each year based on the aggregated text.
    """
    # Ensure NLTK Italian stopwords are downloaded
    try:
        italian_stopwords = set(stopwords.words('italian'))
    except LookupError:
        nltk.download('stopwords')
        italian_stopwords = set(stopwords.words('italian'))

    # Combine with WordCloud's default stopwords
    combined_stopwords = STOPWORDS.union(italian_stopwords)

    # Create a dictionary to hold text per year
    text_per_year = {}

    # Extract year and aggregate text
    for line in all_text.split('\n'):
        try:
            # Assuming each line corresponds to a file's text
            # and that the filename contains the date
            # Since we have concatenated all texts, we'll need to re-extract dates
            # However, to simplify, let's modify 'load_word_counts' to also return texts per date
            pass  # Placeholder
        except Exception as e:
            print(f"Error processing line: {e}")

    # Instead of above, it's better to modify 'load_word_counts' to return texts per year
    # Update 'load_word_counts' to return a dictionary of year to text
    # To avoid confusion, let's adjust the 'load_word_counts' function

def generate_wordclouds_per_year_updated(folder_path):
    """
    Generates and saves a word cloud for each year based on the aggregated text.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder_path} is not a valid directory.")

    # Ensure NLTK Italian stopwords are downloaded
    try:
        italian_stopwords = set(stopwords.words('italian'))
    except LookupError:
        nltk.download('stopwords')
        italian_stopwords = set(stopwords.words('italian'))

    # Combine with WordCloud's default stopwords
    combined_stopwords = STOPWORDS.union(italian_stopwords)

    # Dictionary to hold text per year
    text_per_year = {}

    for file in folder.glob('*.txt'):
        try:
            date_str = extract_date(file.name)
            date = pd.to_datetime(date_str)
            year = date.year
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                if year in text_per_year:
                    text_per_year[year] += ' ' + text
                else:
                    text_per_year[year] = text
        except ValueError as e:
            print(f"Skipping file: {e}")

    # Generate word cloud for each year
    for year, text in text_per_year.items():
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=combined_stopwords,
            collocations=False,
            max_words=200,
            # lang='it'  # Specify Italian language
        ).generate(text)

        # Plot the word cloud
        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {year}', fontsize=20)
        plt.tight_layout(pad=0)
        # Save the word cloud image
        output_image = folder / f'wordcloud_{year}.png'
        plt.savefig(output_image)
        plt.close()
        print(f"Word cloud for {year} saved as {output_image}")

def main():
    # Specify the folder containing the text files
    folder_path = '/Users/vigji/Desktop/sando-data/whispered_backup'  # <-- Replace with your folder path

    # Load word counts and concatenated text
    df, concatenated_text = load_word_counts(folder_path)
    print(df)

    # Save the concatenated text
    save_concatenated_text(folder_path, concatenated_text)

    # Plot the word counts
    plot_word_counts(df)

    # Generate word clouds per year
    generate_wordclouds_per_year_updated(folder_path)

if __name__ == "__main__":
    main()
