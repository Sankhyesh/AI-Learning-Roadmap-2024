# Natural Language Processing (NLP) - Lecture 3: Text Preprocessing Techniques

## Overview

This lecture covers the essential text preprocessing techniques in NLP, which form the crucial second step in the NLP pipeline after data acquisition. The instructor emphasizes that preprocessing directly impacts model performance, as inconsistent or noisy text can confuse ML algorithms and reduce accuracy.

**Core Principle**: Not all preprocessing steps are universally applicable - the choice depends on your specific problem and dataset characteristics.

## Text Preprocessing Techniques Covered

### 1. Lowercasing
**Purpose**: Standardize text by converting all characters to lowercase

**Why it matters**: 
- Prevents models from treating "Basically" and "basically" as different words
- Reduces unnecessary complexity in the vocabulary
- Essential first step in most NLP pipelines

**Implementation**:
```python
# For a single string
text = text.lower()

# For pandas DataFrame
df['reviews'] = df['reviews'].str.lower()
```

**Example**: "I am an INDIAN" → "i am an indian"

### 2. HTML Tag Removal
**Purpose**: Strip HTML markup from web-scraped data

**Why it matters**:
- HTML tags carry no semantic meaning for text analysis
- Tags like `<br>`, `<p>`, `<body>` add noise to the data
- Common issue when scraping data from websites

**Implementation using regex**:
```python
import re

def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return re.sub(pattern, '', text)
```

**Pattern explanation**: `<.*?>` matches any content between < and >

### 3. URL Removal
**Purpose**: Eliminate web links from text data

**Common in**: 
- Social media datasets (Twitter, WhatsApp)
- Chat applications
- Forum discussions

**Implementation**:
```python
def remove_urls(text):
    # Handles http://, https://, www. variations
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return re.sub(pattern, '', text)
```

### 4. Punctuation Removal
**Purpose**: Eliminate punctuation marks that can create tokenization issues

**Two main problems addressed**:
1. Punctuation treated as separate tokens (wasteful)
2. Words with punctuation treated differently ("hello" vs "hello!")

**Fast implementation** (3x faster than loop-based):
```python
import string

def remove_punctuation_fast(text):
    exclude = string.punctuation
    return text.translate(str.maketrans('', '', exclude))
```

**Note**: Keep punctuation for certain tasks like Part-of-Speech tagging

### 5. Chat Word Treatment
**Purpose**: Expand chat abbreviations to their full forms

**Examples**:
- "gn" → "good night"
- "asap" → "as soon as possible"  
- "imho" → "in my honest opinion"

**Implementation**:
```python
def chat_conversion(text):
    chat_words = {
        'gn': 'good night',
        'asap': 'as soon as possible',
        'imho': 'in my honest opinion',
        'fyi': 'for your information'
    }
    
    words = text.split()
    expanded = [chat_words.get(word, word) for word in words]
    return ' '.join(expanded)
```

### 6. Spelling Correction
**Purpose**: Fix common spelling mistakes in text data

**When to use**:
- User-generated content
- Social media data
- Chat applications

**Implementation using TextBlob**:
```python
from textblob import TextBlob

def correct_spelling(text):
    blob = TextBlob(text)
    return blob.correct()
```

**Example**: "plese read this notbok" → "please read this notebook"

### 7. Stop Words Removal
**Purpose**: Remove common words that don't contribute to meaning

**Stop words include**: "the", "is", "at", "which", "on", etc.

**Implementation using NLTK**:
```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered = [w for w in words if w not in stop_words]
    return ' '.join(filtered)
```

**Note**: Don't remove stop words for Part-of-Speech tagging tasks

### 8. Emoji Handling
**Purpose**: Either remove emojis or replace them with text descriptions

**Two approaches**:
1. Remove emojis completely
2. Replace with meaningful text (😄 → "smile")

**Remove emojis**:
```python
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
```

**Replace with text using emoji library**:
```python
import emoji

def emoji_to_text(text):
    return emoji.demojize(text)
```

### 9. Tokenization
**Definition**: Breaking text into smaller units (tokens)

**Types**:
- **Word tokenization**: "I am Indian" → ["I", "am", "Indian"]
- **Sentence tokenization**: Splitting paragraphs into sentences

**Key challenges**:
- Handling contractions (let's, I'm)
- Dealing with abbreviations (U.S., Ph.D.)
- Managing punctuation attached to words
- Special cases (emails, URLs, numbers with units)

**Best practices**:

1. **Basic approach** (limited):
```python
tokens = text.split()  # Works for simple cases
```

2. **Using NLTK** (better):
```python
from nltk.tokenize import word_tokenize, sent_tokenize

word_tokens = word_tokenize(text)
sentence_tokens = sent_tokenize(text)
```

3. **Using spaCy** (recommended):
```python
import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp(text)
tokens = [token.text for token in doc]
```

**Why spaCy is preferred**: Better handles edge cases like "Ph.D.", "10.5km", email addresses

### 10. Stemming
**Definition**: Reducing words to their root form by removing suffixes/prefixes

**Key characteristics**:
- Fast algorithm-based approach
- Root form may not be a valid English word
- Example: "probably" → "probabl"

**Implementation using Porter Stemmer**:
```python
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def stem_words(text):
    words = text.split()
    stemmed = [ps.stem(word) for word in words]
    return ' '.join(stemmed)
```

**When to use**: When speed matters and output isn't shown to users

### 11. Lemmatization
**Definition**: Reducing words to their dictionary form (lemma)

**Key characteristics**:
- Slower but more accurate than stemming
- Always returns valid words
- Uses linguistic knowledge and dictionaries (WordNet)
- Example: "better" → "good", "running" → "run"

**Implementation**:
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    words = text.split()
    # Can specify POS: lemmatizer.lemmatize(word, pos='v')
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)
```

**When to use**: When output quality matters or results are user-facing

## Complete Text Preprocessing Pipeline

```python
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove HTML tags
    text = remove_html_tags(text)
    
    # 3. Remove URLs
    text = remove_urls(text)
    
    # 4. Expand chat words
    text = chat_conversion(text)
    
    # 5. Correct spelling (optional - can be slow)
    # text = correct_spelling(text)
    
    # 6. Remove punctuation
    text = remove_punctuation_fast(text)
    
    # 7. Remove stop words
    text = remove_stopwords(text)
    
    # 8. Tokenization (using preferred method)
    tokens = word_tokenize(text)
    
    # 9. Stemming or Lemmatization (choose one)
    # tokens = [ps.stem(token) for token in tokens]  # Stemming
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatization
    
    return ' '.join(tokens)
```

## Practical Datasets Used

1. **IMDB 50K Movie Reviews**: For sentiment analysis examples
2. **Twitter Hate Speech Dataset**: For social media preprocessing
3. **Custom movie dataset from TMDB API**: For the assignment

## Libraries and Tools

### Essential Libraries:
- **NLTK**: Natural Language Toolkit - comprehensive NLP library
- **spaCy**: Industrial-strength NLP with better tokenization
- **TextBlob**: Simple API for common NLP tasks
- **pandas**: For dataframe operations
- **re (regex)**: For pattern matching and text manipulation

### Performance Tips:
- Use vectorized operations with pandas for large datasets
- Prefer translate() over loops for punctuation removal
- spaCy tokenizer handles edge cases better than split()
- Consider parallel processing for very large datasets

## Assignment: Multi-class Classification Dataset

**Objective**: Create and preprocess a movie genre classification dataset

**Steps**:
1. **Data Collection**:
   - Use TMDB API to fetch 9400+ movies
   - Extract: movie name, description, genre(s)
   - Handle pagination (471 pages, 20 movies/page)

2. **Preprocessing Pipeline**:
   - Apply all preprocessing techniques learned
   - Focus on the description column
   - Save processed dataset for future use

**API Endpoints**:
- Top-rated movies: `[TMDB API endpoint provided]`
- Genre details: `[TMDB API endpoint for genres]`

## Key Takeaways

1. **Preprocessing is problem-specific**: Not all techniques apply to every dataset
2. **Order matters**: Generally follow: lowercase → remove noise → tokenize → normalize
3. **Performance considerations**: Choose fast methods for large datasets
4. **Library selection**: spaCy for accuracy, NLTK for flexibility, TextBlob for simplicity
5. **Validation**: Always inspect preprocessed output to catch edge cases

## Next Steps

- Implement the complete preprocessing pipeline on the assignment dataset
- Experiment with different preprocessing combinations
- Move to feature engineering techniques (Bag of Words, TF-IDF)
- Build your first NLP model using preprocessed data

## Visual Aid Suggestions

- **Flowchart**: Illustrating the complete preprocessing pipeline with decision points for optional steps
- **Before/After table**: Showing text transformation at each preprocessing stage
- **Comparison chart**: Stemming vs Lemmatization outputs for common words
- **Performance graph**: Comparing execution times of different tokenization methods

## Learning Prompts for Deeper Reflection

1. **How might preprocessing requirements differ between formal documents (academic papers) versus informal text (tweets)?** Consider the impact of abbreviations, emojis, and grammatical structure.

2. **What preprocessing steps might actually harm model performance for certain NLP tasks?** Think about tasks where punctuation or capitalization carry important meaning.

## Code Resources

- Complete notebook with all examples: [Kaggle notebook link provided]
- Video tutorial: [YouTube link provided]
- GitHub repository with preprocessing templates: [To be created]

[End of Notes]