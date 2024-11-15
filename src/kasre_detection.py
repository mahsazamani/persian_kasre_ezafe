import re
import string
import demoji
from hazm import Normalizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class KasreEzafe:
    def __init__(self, model_path="models/bert_finetuned",device="cpu"):
        self.device=device
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.normalizer = Normalizer()
        self.remove_digits = str.maketrans('', '', string.digits + '۰۱۲۳۴۵۶۷۸۹' + '0123456789')
        self.exclude = set(string.punctuation) - {".", "،"}
        self.id2label = {2:'I-WORD',1:'B-WORD',0:'O'}

    def remove_em(self, text):
        """Remove emojis from the text."""
        return demoji.replace(text, '')

    def deLinkify(self, text):
        """Remove links from the text."""
        link_pattern = re.compile(r"((https?|ftp)://)?(?<!@)(([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})[-\w@:%_.+/~#?=&]*")
        return link_pattern.sub('', text)

    def deEmojify(self, text):
        """Remove emojis from the text."""
        emoji_pattern = re.compile(
            "[" 
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)

    def remove_dots(self, text):
        """Remove specific punctuation marks from the text."""
        for punct in ["…", "«", "»"]:
            text = text.replace(punct, '')
        return text.strip()

    def preprocessing(self, input_text):
        """Normalize input text using a series of cleaning functions."""
        input_text = re.sub("[A-Za-z]", "", str(input_text))
        input_text = self.deEmojify(input_text)
        input_text = self.deLinkify(input_text)
        input_text = self.remove_em(input_text)
        input_text = ''.join(ch for ch in input_text if ch not in self.exclude)
        input_text = input_text.translate(self.remove_digits)
        input_text = self.remove_dots(input_text)
        input_text = re.sub("چهل سالگی انقلاب", "چهلسالگیانقلاب", input_text)
        
        # Normalize repeated special character (e.g., ?)
        special_char = "؟"
        pattern = f'({re.escape(special_char)})\\1+'
        input_text = re.sub(pattern, r'\1', input_text)
        
        return self.normalizer.normalize(input_text)


    def predict(self, input_text,MAX_LEN=128):
        input_text = self.preprocessing(input_text)
        inputs = self.tokenizer(input_text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")

        ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)

        outputs = self.model(ids, mask)
        logits = outputs[0]

        active_logits = logits.view(-1, self.model.num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)

        tokens = self.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [self.id2label[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) 

        word_level_predictions = []
        for pair in wp_preds:
            if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
                # skip prediction
                continue
            else:
                word_level_predictions.append(pair[1])

        str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")
        #print(str_rep)
        #print(word_level_predictions)
        return wp_preds,word_level_predictions


