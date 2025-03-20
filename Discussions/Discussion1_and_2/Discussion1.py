# Import necessary libraries
import nltk
from nltk.corpus import brown

# Retrieve the corpus 
nltk.download('brown')

# Access the categories in the Brown Corpus
categories = brown.categories()
print("Categories in the Brown Corpus:")
print(categories)

# Example: Let's access words and sentences from a specific category
news_words = brown.words(categories='news')
news_sentences = brown.sents(categories='news')

# Display some words and sentences
print("\nSample Words from 'News' Category:")
print(news_words[:20])

print("\nSample Sentences from 'News' Category:")
print(news_sentences[:3])

# Perform a simple frequency distribution analysis
from nltk.probability import FreqDist

fdist = FreqDist(news_words)
print("\nMost Common Words in 'News' Category:")
print(fdist.most_common(10))

# Plot the frequency distribution
import matplotlib.pyplot as plt

fdist.plot(30, title="Word Frequency in 'News' Category (Top 30)")

#%% Let's go a bit deeper in our analaysis and use POS (Part of Speech) tagging

# Import libraries
import nltk
from nltk.corpus import brown
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# Download required resources
nltk.download('averaged_perceptron_tagger')

# Let's have a new_words variable 
news_words = brown.words(categories='news')

# Step 1: Frequency Distribution of Words
fdist = FreqDist(news_words)

print("\nTop 10 Most Frequent Words in 'News':")
print(fdist.most_common(10))

# Step 2: Collocation Analysis (Bigrams)
bigram_finder = BigramCollocationFinder.from_words(news_words)
bigram_finder.apply_freq_filter(5)  # Consider bigrams with frequency > 5
top_bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)

print("\nTop 10 Bigrams in 'News':")
print(top_bigrams)

# Step 3: POS Tagging
news_sentences = brown.sents(categories='news')
sample_sentence = news_sentences[0]  # Get the first sentence

print("\nSample Sentence from 'News':")
print(" ".join(sample_sentence))

# Perform POS tagging
tagged_sentence = nltk.pos_tag(sample_sentence)
print("\nPOS Tagged Sentence:")
print(tagged_sentence)

# Step 4: Frequency of Nouns, Verbs, and Adjectives
tags = [tag for word, tag in tagged_sentence]
fdist_tags = FreqDist(tags)

print("\nTop POS Tags in the Sample Sentence:")
print(fdist_tags.most_common())

# Step 5: Advanced Visualization: Word Length vs. Frequency
import matplotlib.pyplot as plt

# Word Length Analysis
word_lengths = [len(word) for word in news_words if word.isalpha()]
fdist_lengths = FreqDist(word_lengths)

plt.figure(figsize=(10, 5))
fdist_lengths.plot(15, title="Word Length Frequency Distribution (News Category)")
plt.show()

# %%
