import keras
import json
import os.path
import urllib.request

from focal_loss import BinaryFocalLoss
from bert import Bert

class Education2Skill:
    def __init__(self, model_filename, lpk_index_map_filename, top_prediction_count=3):
        def download_model(url):
            if not os.path.isfile(model_filename):
                with open(model_filename, 'wb') as fp:
                    with urllib.request.urlopen(url) as f:
                        fp.write(f.read())

        download_model('https://github.com/Stratosfera/education2skill/releases/download/v1/finaldnau1000withoutweightsseq.h5')
    
        self.bert = Bert()        
        self.model = keras.models.load_model(model_filename)
        self.top_prediction_count = top_prediction_count

        with open('./models/lpkindexmap.json', 'r') as f:
            self.lpkindexmap = json.load(f)

    def skills_from_single_description(self, description):
        embedding = self.bert.make_single_embedding(description)
        return self.skills_from_single_embedding(embedding)

    def skills_from_descriptions(self, descriptions_dataframe):
        embeddings = self.bert.make_embeddings(descriptions_dataframe)
        return self.skills_from_embeddings(embeddings)
    
    def skills_from_single_embedding(self, embedding):
        raw_predictions = self.model.predict(embedding)
        prediction_map = {}

        for level, raw_prediction in enumerate(raw_predictions):
            sorted_prediction_indeces = raw_prediction[0].argsort()[-self.top_prediction_count:][::-1]
            prediction_map[level + 1] = [(index, raw_prediction[0][index]) for index in sorted_prediction_indeces]

        for level, predictions in prediction_map.items():
            for index, probability in predictions:
                yield level, index, probability, self.lpkindexmap[str(level)][index]

    def skills_from_embeddings(self, embeddings):    
        for embedding in embeddings:
            predicted_occupations = self.skills_from_single_embedding(embedding)
            yield embedding, predicted_occupations


    
    
