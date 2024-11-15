
import pandas as pd
from hazm import POSTagger, word_tokenize
from kasre_detection import KasreEzafe

# Initialize the POS tagger
tagger = POSTagger(model='models/pos_tagger.model')

def process_text(text):
    if not isinstance(text, str):
        return ''  # Return an empty string if text is not a valid string

    text_tags = tagger.tag(word_tokenize(text))

    # Filter words with tags 'NOUN,EZ' or 'ADJ,EZ' or 'ADP,EZ'
    selected_words = [word for word, tag in text_tags if tag in ['NOUN,EZ', 'ADJ,EZ', 'ADP,EZ']]

    return '،'.join(selected_words)

if __name__ == "__main__":
    ke_obj = KasreEzafe()
    # df = pd.read_csv('/home/zamani/Documents/neshan/data/original_twitter_data.txt')
    # raw_data = df[:50000]
    # raw_data.to_csv('/home/zamani/Documents/neshan/data/raw_data.csv',index=False)
    
    df = pd.read_csv('/home/zamani/Documents/neshan/data/raw_data.csv')
    #df.drop(columns=["Letter"],inplace=True)
    
    df['text_query'] = df['text_query'].apply(ke_obj.preprocessing)
    df['selected_words'] = df['text_query'].apply(process_text)
    #df.to_csv('/home/zamani/Documents/neshan/data/lscp-50000.csv')

