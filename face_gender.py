import cv2
import glob
import numpy as np
import random
#initilaizing fisherface recognizer

fisherface = cv2.face.FisherFaceRecognizer_create()

def getfiles(gender, training_size):
    #loading the dataset
    file = glob.glob("D:\\cropped_faces\\{0}\\*" .format(gender))
    random.shuffle(file)
    train = file[:int(len(file) * training_size)]
    predict = file[-int(len(file) * (1 - training_size)):]
    return train, predict

def make_sets(): #creating lists
    train_data = []
    train_labels = []
    predict_data = []
    predict_label = []
    for gender in genders:
        training_set, prediction_set = getfiles(gender, 0.8) #getting first 805 of files

        for object in training_set:
            img = cv2.imread(object, 0)#reading the object image
            face2 = cv2.resize(img, (350, 350)) #resizing the image

            train_data.append(face2)
            train_labels.append(genders.index(gender))

        for object in prediction_set:
            object = cv2.imread(object, 0) #reading the object
            face2 = cv2.resize(object, (350, 350)) #resizing the object

            predict_data.append(face2)
            predict_label.append(genders.index(gender))

    return train_data, train_labels, predict_data, predict_label


def run_recognizer():
    data_training, labels_training, data_prediction, labels_predictions = make_sets()

    print("size of the training set is", len(labels_training), "images")

#training the daraset
    fisherface.train(data_training, np.asarray(labels_training))


    positive = 0
    for id, img in enumerate(data_prediction):
        if (fisherface.predict(img)[0] == labels_predictions[id]):
            positive += 1

    percent = (positive * 100) / len(data_prediction)

    return positive, percent

if __name__ == '__main__':
    genders = ["female", "male"]

    positive, percent = run_recognizer()
    print("Processed ", positive, " data correctly")
    print("Got ", percent, " accuracy")

#writing the training data
    fisherface.write('D:\\models\gender_classifier_model.xml')
