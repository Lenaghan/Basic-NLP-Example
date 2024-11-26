import csv
import os

# Function to load bold words from a text file
def load_word_list(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]
    
# Function to wrap bold words with <strong> tags
def emphasize_words(text, bold_words, italics_words):
    for word in bold_words:
        text = text.replace(word, f"<strong>{word}</strong>")
    for word in italics_words:
        text = text.replace(word, f"<em>{word}</em>")
    return text
    
# Function to add HTML tags
def add_html_tags(input_file, output_file, bold_words_list, italics_word_list):

    # Check if the output file exists, if not, create it
    if not os.path.exists(output_file):
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Column1', 'Column2'])  # add default headers
    
    # Load emphasize words list from files
    bold_words = load_word_list(bold_words_list)
    italics_words = load_word_list(italics_word_list)

    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        unmodified_file = csv.reader(infile)
        header = next(unmodified_file)  # Read the header row if present

        # Open the output file
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)

            # Write the header row
            writer.writerow(header)

            # Process each row
            for row in unmodified_file:
                if len(row) > 1:  # Ensure the row has at least two columns
                    row[1] = emphasize_words(f"<h2>{row[1]}</h2>", bold_words, italics_words)  # Add tags
                writer.writerow(row)

# Define file paths and bold words
input_csv = './data/clickbait_data.csv'
output_csv = './data/data_cb.csv'
bold_words_file = './data/bold.txt' 
italics_word_file = './data/italics.txt'

# Add HTML tags to the headline column
add_html_tags(input_csv, output_csv, bold_words_file, italics_word_file)