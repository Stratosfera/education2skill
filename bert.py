import numpy as np
import pandas as pd
import torch
import transformers
import warnings
warnings.filterwarnings('ignore')

class Bert:
    def __init__(self):
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = transformers.BertModel.from_pretrained('bert-base-uncased')

    def make_embeddings(self, inputs_dataframe, batch_size=100):
        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]
                
        for k, input_row in enumerate(batch(inputs_dataframe, batch_size)):
            tokenized_text = input_row.apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True, max_length = 512, truncation=True)))
            max_len = 0
            for i in tokenized_text.values:
                if len(i) > max_len:
                    max_len = len(i)
        
            padded_input = np.array([i + [0]*(max_len-len(i)) for i in tokenized_text.values])
            attention_mask = np.where(padded_input != 0, 1, 0)
            input_ids = torch.tensor(padded_input).long()  
            attention_mask = torch.tensor(np.array(attention_mask)).long()
        
            with torch.no_grad():
                last_hidden_states = self.model(input_ids, attention_mask=attention_mask)
            
            yield last_hidden_states[0][:,0,:].numpy()
    
    def make_single_embedding(self, input_text):
        for embedding in self.make_embeddings(pd.Series([input_text], dtype=pd.StringDtype()), 1):
            return embedding

