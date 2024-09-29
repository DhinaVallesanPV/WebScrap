import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

def extract_article(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Parse HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find article title
    title = soup.find('title').get_text()

    # Find article text
    article_text = ""
    article_body = soup.find('body')
    for paragraph in article_body.find_all('p'):
        article_text += paragraph.get_text() + "\n"

    return title, article_text


def main():
    # Read URLs from input Excel file
    input_df = pd.read_excel('Input.xlsx')

    # Initialize a list to store extracted data
    extracted_data = []

    # Create a directory to store text files if it doesn't exist
    if not os.path.exists('extracted_articles'):
        os.makedirs('extracted_articles')

    # Iterate over each URL in the input DataFrame
    for index, row in input_df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']

        # Extract article from URL
        title, article_text = extract_article(url)

        # Save article text to a text file
        file_path = os.path.join('extracted_articles', f'{url_id}.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(article_text)

        # Append extracted data to the list
        extracted_data.append({'URL_ID': url_id, 'URL': url, 'Title': title, 'Article_text': article_text})

    # Convert extracted data to a DataFrame
    extracted_df = pd.DataFrame(extracted_data)

    # Save extracted data to a single CSV file
    extracted_df.to_csv('extracted_articles.csv', index=False)


if __name__ == '__main__':
    main()
