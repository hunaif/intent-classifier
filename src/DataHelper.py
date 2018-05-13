import spacy
import json
import io
import numpy as np
print ("Loading spacy model")
spacy_model = spacy.load("en_core_web_md")
print ("Done loading spacy model")
from Constants import INDEX_TO_LABEL_FILE_PATH,DATA_PATH

class DataHelper:
    index_to_label_dict = None
    pass

    @classmethod
    def get_index_label_dict(cls):
        if cls.index_to_label_dict is None:
            try:
                with open(INDEX_TO_LABEL_FILE_PATH) as json_data:
                    cls.index_to_label_dict = json.load(json_data)
            except:
                print("Unable to read index to index to label dict")
        return cls.index_to_label_dict

    @classmethod
    def get_data_and_labels(cls):
        with io.open(DATA_PATH,"r",encoding="utf-8") as f:
            x_data = []
            y_data = []
            for line in f:
                if(line.__contains__(",,,") and len(line.split(",,,")) == 2):
                    line_splitted = line.split(",,,")
                    x = line_splitted[0].strip()
                    y = line_splitted[1].strip()
                    x_lemmatized = DataHelper.lemmatize(x)
                    x_data.append(np.array(spacy_model(x_lemmatized).vector))
                    y_data.append(y)
        return x_data,y_data

    @classmethod
    def lemmatize(cls,text):
        sent = []
        doc = spacy_model(text)
        for word in doc:
            sent.append(word.lemma_)
        return " ".join(sent)

    @classmethod
    def save_index_to_label_mapping(cls,label_indexes,le):
        index_to_label_dict = {}
        label_indexes_list = list(set(label_indexes))
        labels_list = le.inverse_transform(label_indexes_list)
        for i in range(len(labels_list)):
            index_to_label_dict[str(label_indexes_list[i])] = labels_list[i]

        with open(INDEX_TO_LABEL_FILE_PATH, 'w') as outfile:
            json.dump(index_to_label_dict, outfile)

if __name__ == "__main__":
    lem = DataHelper.lemmatize("who are you")
    print(lem)


