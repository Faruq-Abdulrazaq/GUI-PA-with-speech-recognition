from PyQt5 import QtCore, QtGui, QtWidgets
import random
import nltk
import speech_recognition as sr
import pyttsx3 as ttx
import sys
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras.models import Sequential

import tflearn
import numpy as np
import random
import json
import pickle

speaker = ttx.init()
speaker.setProperty('rate', 150)
stemmer = LancasterStemmer()
classifier = Sequential()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        nltk.download('punkt')
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(580, 507)
        MainWindow.setStyleSheet("background-color: rgb(0, 255, 127);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(19, 19, 541, 461))
        self.frame.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(9, 10, 521, 441))
        self.frame_2.setStyleSheet("background-color: rgb(0, 255, 127);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.chatBrowser = QtWidgets.QTextBrowser(self.frame_2)
        self.chatBrowser.setGeometry(QtCore.QRect(35, 20, 461, 381))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.chatBrowser.setFont(font)
        self.chatBrowser.setStyleSheet("background-color: rgb(0, 255, 127);\n"
                                       "color: rgb(255, 255, 255);")
        self.chatBrowser.setObjectName("chatBrowser")
        self.pushButton = QtWidgets.QPushButton(self.frame_2)
        self.pushButton.setGeometry(QtCore.QRect(230, 410, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                      "border-color: rgb(0, 255, 127);")
        self.pushButton.setObjectName("pushButton")
        self.introlabel = QtWidgets.QLabel(self.frame)
        self.introlabel.setGeometry(QtCore.QRect(110, 0, 351, 20))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.introlabel.setFont(font)
        self.introlabel.setObjectName("introlabel")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.chatBrowser.setHtml(_translate("MainWindow",
                                            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            "</style></head><body style=\" font-family:\'Segoe UI\'; font-size:10pt; font-weight:600; font-style:italic;\">\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Bot: fish</p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">me: yes</p>\n"
                                            "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "Chat"))
        self.chatBrowser.setText(" ")
        self.introlabel.setText(_translate("MainWindow", "Speech recognition and voice assistance"))
        self.chatBrowser.setText(" START ")
        self.pushButton.clicked.connect(self.run)

    def bag_of_words(self, s, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1
        return np.array(bag)

    def get_audio(self):
        r = sr.Recognizer()
        print("reach1")
        with sr.Microphone() as source:
            print("reach2")
            audio = r.listen(source)
            print("reach3")
            said = ""

        try:
            print("reach4")
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            print("Exception: " + str(e))

        return said.lower()

    def make_note(self):
        speaker.say("What do want to note down?")
        speaker.runAndWait()
        done =  False
        while not done:
            note = self.get_audio()
            speaker.say("Choose a file name")
            speaker.runAndWait()
            filename = self.get_audio()
            with open(filename + 'txt', 'w') as w:
                w.write(note)
                done = True
                speaker.say(f"I have created the note {filename}")
                speaker.runAndWait()


    def run(self):
        self.pushButton.clicked.connect(self.run)
        with open("intents.json", "rb") as json_file:
            data = json.load(json_file)

        with open("data.pickle", "rb") as f:
            self.words, labels, training, output = pickle.load(f)

        net = tflearn.input_data(shape=[None, len(training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
        net = tflearn.regression(net)

        model = tflearn.DNN(net, tensorboard_verbose=0)

        model.load("model.tflearn")
        while True:
            me = ""
            self.chat = self.get_audio()
            self.chatBrowser.append(f"Me: {self.chat}")
            if  "make a note" in self.chat or "note this down" in self.chat  or  "remind me this" in self.chat:
                self.make_note()
            else:
                results = model.predict([self.bag_of_words(self.chat, self.words)])[0]
                results_index = np.argmax(results)
                response = labels[results_index]
                if results[results_index] > 0.7:
                    for tg in data["intents"]:
                        if tg['tag'] == response:
                            responses = tg['responses']
                            reply = random.choice(responses)
                            self.chatBrowser.append(f"Bot: {reply}")
                            speaker.say(reply)
                            speaker.runAndWait()

                else:
                    speaker.say("I don't understand!")
                    speaker.runAndWait()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
