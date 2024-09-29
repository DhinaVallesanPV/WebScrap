import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import cmudict

def syllable_count(word):
    if word.lower() in d:
        return max([len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]])
    else:
        return 0

def compute_positive_score(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['pos']

def compute_negative_score(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['neg']

def compute_polarity_score(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['compound']

def compute_subjectivity_score(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['compound']

def compute_avg_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    total_words = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences)
    total_sentences = len(sentences)
    return total_words / total_sentences

def compute_percentage_complex_words(text):
    num_words = len(nltk.word_tokenize(text))
    num_complex_words = sum(1 for word in nltk.word_tokenize(text) if syllable_count(word) > 2)
    return (num_complex_words / num_words) * 100

def compute_fog_index(avg_sentence_length, percentage_complex_words):
    return 0.4 * (avg_sentence_length + percentage_complex_words)

def compute_avg_words_per_sentence(text):
    sentences = nltk.sent_tokenize(text)
    total_words = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences)
    total_sentences = len(sentences)
    return total_words / total_sentences

def compute_complex_word_count(text):
    return sum(1 for word in nltk.word_tokenize(text) if syllable_count(word) > 2)

def compute_word_count(text):
    return len(nltk.word_tokenize(text))

def compute_syllables_per_word(text):
    num_words = len(nltk.word_tokenize(text))
    total_syllables = sum(syllable_count(word) for word in nltk.word_tokenize(text))
    return total_syllables / num_words

def compute_personal_pronouns(text):
    personal_pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs']
    return sum(1 for word in nltk.word_tokenize(text.lower()) if word in personal_pronouns)

def compute_avg_word_length(text):
    total_word_length = sum(len(word) for word in nltk.word_tokenize(text))
    num_words = len(nltk.word_tokenize(text))
    return total_word_length / num_words

def main():
    # Read extracted article text from text files
    # Assume df_articles contains the URL_IDs and article texts
    df_articles = pd.read_csv('extracted_articles.csv')

    # Initialize CMU Pronouncing Dictionary
    global d
    d = cmudict.dict()

    # Compute variables for each article
    df_articles['URL_ID'] = df_articles['URL_ID']
    df_articles['URL'] = df_articles['URL']
    df_articles['POSITIVE_SCORE'] = df_articles['Article_text'].apply(compute_positive_score)
    df_articles['NEGATIVE_SCORE'] = df_articles['Article_text'].apply(compute_negative_score)
    df_articles['POLARITY_SCORE'] = df_articles['Article_text'].apply(compute_polarity_score)
    df_articles['SUBJECTIVITY_SCORE'] = df_articles['Article_text'].apply(compute_subjectivity_score)
    df_articles['AVG_SENTENCE_LENGTH'] = df_articles['Article_text'].apply(compute_avg_sentence_length)
    df_articles['PERCENTAGE_COMPLEX_WORDS'] = df_articles['Article_text'].apply(compute_percentage_complex_words)
    df_articles['FOG_INDEX'] = df_articles.apply(lambda row: compute_fog_index(row['AVG_SENTENCE_LENGTH'], row['PERCENTAGE_COMPLEX_WORDS']), axis=1)
    df_articles['AVG_WORDS_PER_SENTENCE'] = df_articles['Article_text'].apply(compute_avg_words_per_sentence)
    df_articles['COMPLEX_WORD_COUNT'] = df_articles['Article_text'].apply(compute_complex_word_count)
    df_articles['WORD_COUNT'] = df_articles['Article_text'].apply(compute_word_count)
    df_articles['SYLLABLES_PER_WORD'] = df_articles['Article_text'].apply(compute_syllables_per_word)
    df_articles['PERSONAL_PRONOUNS'] = df_articles['Article_text'].apply(compute_personal_pronouns)
    df_articles['AVG_WORD_LENGTH'] = df_articles['Article_text'].apply(compute_avg_word_length)

    # Save results to a file
    df_articles.to_csv('output_variables.csv', index=False, columns=['URL_ID', 'URL', 'POSITIVE_SCORE', 'NEGATIVE_SCORE', 'POLARITY_SCORE', 'SUBJECTIVITY_SCORE', 'AVG_SENTENCE_LENGTH', 'PERCENTAGE_COMPLEX_WORDS', 'FOG_INDEX', 'AVG_WORDS_PER_SENTENCE', 'COMPLEX_WORD_COUNT', 'WORD_COUNT', 'SYLLABLES_PER_WORD', 'PERSONAL_PRONOUNS', 'AVG_WORD_LENGTH'])

if __name__ == '__main__':
    main()
