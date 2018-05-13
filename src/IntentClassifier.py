from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle
from DataHelper import *
from Constants import MODEL_PATH
class IntentClassifier:
    classifier = None

    @classmethod
    def get_svm_config(cls):
        config = {
            "kernel": "linear",
            "C": [1,2,5,10,20,100]
        }
        return config

    @classmethod
    def train(cls):
        X, labels = DataHelper.get_data_and_labels()
        X, labels = shuffle(X,labels)
        le = LabelEncoder()
        y = le.fit_transform(labels)

        DataHelper.save_index_to_label_mapping(y,le)

        sklearn_config = IntentClassifier.get_svm_config()
        C = sklearn_config.get("C")
        kernel = sklearn_config.get("kernel")

        tuned_parameters = [{"C": C, "kernel": [str(kernel)]}]

        clf = GridSearchCV(SVC(probability=True, class_weight='balanced',decision_function_shape='ovr'),
                                param_grid=tuned_parameters,
                                cv=5, scoring='f1_weighted', verbose=1)
        clf.fit(X, y)
        print("best params : {} , best F1: {}".format(clf.best_params_,clf.best_score_))
        f = open(MODEL_PATH, 'wb')
        pickle.dump(clf, f)
        f.close()
        return clf

    @classmethod
    def get_classifier(cls):
        if(cls.classifier is None):
            print("Loading classifier")
            try:
                f = open(MODEL_PATH, 'rb')   # 'rb' for reading binary file
                cls.classifier = pickle.load(f)
                print("Loaded classifier")
                f.close()
            except:
                print("Exception in loading classifier")
        return cls.classifier

    @classmethod
    def predict(cls,query):
        model = cls.get_classifier()
        if(model is not None):
            query_lemmatized = DataHelper.lemmatize(str(query))
            prediction = model.predict_proba(np.array(spacy_model(query_lemmatized).vector).reshape(1,300))
            index_to_label_dict = DataHelper.get_index_label_dict()
            label_index = np.argmax(prediction)
            label = index_to_label_dict[str(label_index)]
            class_score = prediction[0][label_index]
            return {"label":label,
                    "probability":class_score}
        else:
            print("Unable to load classifier")


if __name__ == "__main__":
    intent_classifier = IntentClassifier()

    intent_classifier.train()
    # print(intent_classifier.predict("What time are you leaving"))