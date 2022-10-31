
import nltk
#import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('wordnet')  # Download the dictionary for nltk
nltk.download('omw-1.4')  # OMW dictionary
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text='Originally, vegetables were collected from the wild by hunter-gatherers. Vegetables are all plants. Vegetables can be eaten either raw or cooked.'
question='Can you eat vegetables?'

# Create a lemmatiser instance
lemmatiser=WordNetLemmatizer()

# Function to return a list of lemmas for a sentence
def lemma_me(sent):
    sentence_tokens=nltk.word_tokenize(sent.lower())
    pos_tags=nltk.pos_tag(sentence_tokens)
    #print(pos_tags)
    sentence_lemmas=[]
    for token,pos_tag in zip(sentence_tokens,pos_tags):
        if pos_tag[1][0] in ['N','V','A','R']:
            lemma=lemmatiser.lemmatize(token,pos_tag[1][0].lower())
            sentence_lemmas.append(lemma)
    #   else: print(token)
    return sentence_lemmas

# tokenise the sentences and question
sentence_tokens=nltk.sent_tokenize(text+' '+question)

# Now we need the TfidfVectorizer class, and create an instance of that class linked to the lemma_me function defined higher in this notepad. The vectorizer returns coefficients denoting the importance of each word
tv=TfidfVectorizer(tokenizer=lemma_me)

# Get the pos_tags for the text and question
tf=tv.fit_transform(sentence_tokens)
values=cosine_similarity(tf[-1],tf)
# values is a list of similarities, based on lemmas ands pos_tags, between the question and each sentence in the text.  Note the last item in the list is 1, indicating a 100% match between the question and itself.  This list is nested in an outer list so lets flatten it
values_flat=values.flatten()

# Now we sort the list and get the index of the 2nd highest value 
index=values_flat.argsort()[-2]

# Get the coefficient of the best matching sentence and check it is over 50%, then print that sentence to get the answer to our question
coeff=values_flat[index]
if coeff>0.5:
    print(sentence_tokens[index])
else: print("I don't know")

# Vegetables are all plants.
