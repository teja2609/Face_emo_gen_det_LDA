import cv2
import glob
import numpy as np
import random

#initilaizing fisherface recognizer
fisherface = cv2.face.FisherFaceRecognizer_create()

def getfile(emotion, training_size):
    #loading the dataset for training
    file = glob.glob("C:\\Users\haris\PycharmProjects\python-project\data\\raw_emotion\\{0}\\*" .format(emotion))
    random.shuffle(file) #shuffling the data
    train = file[:int(len(file) * training_size)]
    predict = file[-int(len(file) * (1 - training_size)):]
    return train, predict

def make_sets(): #creating lists
    train_data = []
    train_labels = []
    predict_data = []
    predict_labels = []
    for emotion in emotions:
        training_set, prediction_set = getfile(emotion, 0.8) #getting first 80% of files

        for object in training_set:
            img = cv2.imread(object, 0) #reading the image
            face = cv2.resize(img, (350, 350)) #resizng all the images to same sizes
            train_data.append(face)
            train_labels.append(emotions.index(emotion))

        for object in prediction_set:
            object = cv2.imread(object, 0) #reading the image
            face1 = cv2.resize(object, (350, 350)) #resizing the images
            predict_data.append(face1)
            predict_labels.append(emotions.index(emotion))

    return train_data, train_labels, predict_data, predict_labels


def run_recognizer():
    data_training, labels_training, data_prediction, labels_prediction = make_sets()

    print("size of the training set is", len(labels_training), "images")
    fisherface.train(data_training, np.asarray(labels_training)) #training the data usimg the fishferface.train function

    print("size of the prediction set is:", len(labels_prediction), "images")
    positive = 0
    for idx, image in enumerate(data_prediction):
        if (fisherface.predict(image)[0] == labels_prediction[idx]):
            positive += 1

    percent = (positive * 100) / len(labels_prediction)

    return positive, percent

if __name__ == '__main__': #types of emotions
    emotions = ["afraid","angry","disgusted","happy","neutral","sad","surprised"]

    positive, percent = run_recognizer()
    print("handled ", positive, " data correctly")
    print("obtained", percent, " accuracy")

#Writing  the trained dataset
    fisherface.write('D:\\models\emotion_classifier_model.xml')
