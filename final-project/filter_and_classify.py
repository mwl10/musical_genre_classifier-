import sys
from keras.preprocessing.text import Tokenizer
from numpy import array
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import seaborn as sn


# This Python Script generally takes the output from the program that finds the songs that are both in the musiXmatch dataset
# as well as the MSD genre dataset and takes the genre and bag-of-words for each song, formatting them properly for machine
# learning classification

filter = open("./filter3.txt")
vocabulary = open("./vocab.txt")

# make a list out of the given 5000 word vocabulary
vocab = vocabulary.readlines()
vocab_list = vocab[0].split(",")
vocab_list.insert(0,"fix offset")


line = filter.readlines()

# formatting the filter document
song_tags = line[0].split("\\n")
song_tags[0] = "'{}".format(song_tags[0])
song_tags = [s.replace("[", '') for s in song_tags]
song_tags = [s[1:] for s in song_tags]
song_tags = ["{}'".format(s) for s in song_tags]

# getting the genre tags and the bags-of-words the way we want them
new_song_tags = []
for song in song_tags:
    list = song.split(",")
    genre_tag = list[0][1:-1]
    song_lyrics = ""
    for s in list[1:]:
        amount_occur = s[2:-1]
        # now count is 1:2
        vocab_index = int(amount_occur.split(":")[0])
        vocab_index_count = int(amount_occur.split(":")[1])
        word_amount = ""
        # now we want to print the tokens
        while vocab_index_count > 0:
            word_amount += "{} ".format(vocab_list[vocab_index])
            vocab_index_count -= 1
        song_lyrics += word_amount
    genre_tag_and_lyrics = [genre_tag, song_lyrics]
    new_song_tags.append(genre_tag_and_lyrics)

# new_song_tags now has a list of a list of each song's genre tag and bag-of-words lyrics
new_song_tags = new_song_tags[:-1]

# 17495: amount of songs with genre tags and lyrics we have to use
print(len(new_song_tags))

# this method makes a list of all of the songs bag-of-words of a particular genre (from all the songs)
# and splits the list into training and test data sets (75-25)


def genre_grab_split(genre_type):
    genre_list = []
    for song in new_song_tags:
        if song[0] == genre_type:
            genre_list.append(song[1])
    length = len(genre_list)
    train_index = round(.75 * length)
    genre_train_list = genre_list[0:train_index]
    genre_test_list = genre_list[train_index:]
    return genre_train_list, genre_test_list

# definging the training and test sets for each genre
soul_and_reggae_train, soul_and_reggae_test = genre_grab_split('soul and reggae')

hip_hop_train, hip_hop_test = genre_grab_split('hip-hop')

classical_train, classical_test = genre_grab_split('classical')

jazz_and_blues_train, jazz_and_blues_test = genre_grab_split('jazz and blues')


metal_train, metal_test = genre_grab_split('metal')

dance_and_electronica_train, dance_and_electronica_test = genre_grab_split('dance and electronica')

pop_train, pop_test = genre_grab_split('pop')

folk_train, folk_test = genre_grab_split('folk')

punk_train, punk_test = genre_grab_split('punk')

classic_pop_and_rock_train, classic_pop_and_rock_test = genre_grab_split('classic pop and rock')

# combining lists of bags-of-words into training and testing sets
training_x = soul_and_reggae_train + hip_hop_train + classical_train + jazz_and_blues_train + metal_train + dance_and_electronica_train + pop_train + folk_train + punk_train + classic_pop_and_rock_train
testing_x = soul_and_reggae_test + hip_hop_test + classical_test + jazz_and_blues_test + metal_test + dance_and_electronica_test + pop_test + folk_test + punk_test + classic_pop_and_rock_test


print("MACHINE LEARNING NOW ---------------------------------------------------------------------------------------")

# using a tokenizer to make a propr vector representation of the bag-of-words data for the classifier
tokenizer = Tokenizer()
vocab_set = set(vocab_list)
tokenizer.fit_on_texts(training_x)
Xtrain = tokenizer.texts_to_matrix(training_x, mode = 'freq')
print(Xtrain.shape)

Xtest = tokenizer.texts_to_matrix(testing_x, mode = 'freq')
print(Xtest.shape)

# aligning the genre tags to be the same as the bags-of-words index
Ytrain = array(['soul_and_reggae' for _ in range(791)] + ['hip_hop' for _ in range(69)] + ['classical' for _ in range(39)] + ['jazz_and_blues' for _ in range(290)] + ['metal' for _ in range(938)] + ['dance_and_electronica' for _ in range(441)] + ['pop' for _ in range(582)] + ['folk' for _ in range(2902)] + ['punk' for _ in range(766)] + ['classic_pop_and_rock' for _ in range(6303)])
Ytest = array(['soul_and_reggae' for _ in range(264)] + ['hip_hop' for _ in range(23)] + ['classical' for _ in range(13)] + ['jazz_and_blues' for _ in range(97)] + ['metal' for _ in range(313)] + ['dance_and_electronica' for _ in range(147)] + ['pop' for _ in range(194)] + ['folk' for _ in range(967)] + ['punk' for _ in range(255)] + ['classic_pop_and_rock' for _ in range(2101)])

# creating the classifier and varying the number of neighbors

model = KNeighborsClassifier(n_neighbors=20)
# ....... training the model ..........
model.fit(Xtrain, Ytrain)
# ....... predicting on the test data ........
Ypredict = model.predict(Xtest)
# printing the accuracy score
print("Accuracy:", metrics.accuracy_score(Ytest, Ypredict))
print(metrics.classification_report(Ytest, Ypredict))


import matplotlib.pyplot as plt
# allow plots to appear within the notebook

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
accuracy_scores = [ 0.3074988568815729, 0.3904892546867855, 0.403063557384545, 0.3904892546867855, 0.407864654778235, 0.41586648376771834, 0.41838134430727025, 0.4204389574759945, 0.41380887059899407, 0.4231824417009602, 0.4204389574759945, 0.4231824417009602, 0.4224965706447188, 0.42158207590306357, 0.4336991312299954, 0.4357567443987197, 0.43621399176954734, 0.4444444444444444, 0.4444444444444444, 0.45038866026520347]
plt.plot(range(1,21), accuracy_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

# confusion matrix
from sklearn.metrics import plot_confusion_matrix
#class_names = ['soul and reggae', 'hip-hop', 'classical', 'jazz and blues', 'metal', 'dance and electronica', 'pop', 'folk', 'punk', 'classic pop and rock']
cmat = metrics.confusion_matrix(Ytest, Ypredict)
plt.figure(figsize=(6.5,6.5))
sn.heatmap(cmat, annot=True, annot_kws={"size": 13}, fmt='g')
plt.show()

# let now do this without the genres classical, dance and electronica, jazz and blues




training_x = soul_and_reggae_train + hip_hop_train + metal_train + pop_train + folk_train + punk_train + classic_pop_and_rock_train
testing_x = soul_and_reggae_test + hip_hop_test + metal_test + pop_test + folk_test + punk_test + classic_pop_and_rock_test


print("WITHOUT JAZZ, ELEC, ClASS ---------------------------------------------------------------------------------------")

# using a tokenizer to make a propr vector representation of the bag-of-words data for the classifier
tokenizer = Tokenizer()
vocab_set = set(vocab_list)
tokenizer.fit_on_texts(training_x)
Xtrain = tokenizer.texts_to_matrix(training_x, mode = 'freq')
print(Xtrain.shape)

Xtest = tokenizer.texts_to_matrix(testing_x, mode = 'freq')
print(Xtest.shape)

# aligning the genre tags to be the same as the bags-of-words index
Ytrain = array(['soul_and_reggae' for _ in range(791)] + ['hip_hop' for _ in range(69)] + ['metal' for _ in range(938)] + ['pop' for _ in range(582)] + ['folk' for _ in range(2902)] + ['punk' for _ in range(766)] + ['classic_pop_and_rock' for _ in range(6303)])
Ytest = array(['soul_and_reggae' for _ in range(264)] + ['hip_hop' for _ in range(23)] + ['metal' for _ in range(313)] + ['pop' for _ in range(194)] + ['folk' for _ in range(967)] + ['punk' for _ in range(255)] + ['classic_pop_and_rock' for _ in range(2101)])

# creating the classifier and varying the number of neighbors

model = KNeighborsClassifier(n_neighbors=20)
# ....... training the model ..........
model.fit(Xtrain, Ytrain)
# ....... predicting on the test data ........
Ypredict = model.predict(Xtest)
# printing the accuracy score
print("Accuracy:", metrics.accuracy_score(Ytest, Ypredict))
print(metrics.classification_report(Ytest, Ypredict))

# allow plots to appear within the notebook

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
#accuracy_scores = [ 0.3074988568815729, 0.3904892546867855, 0.403063557384545, 0.3904892546867855, 0.407864654778235, 0.41586648376771834, 0.41838134430727025, 0.4204389574759945, 0.41380887059899407, 0.4231824417009602, 0.4204389574759945, 0.4231824417009602, 0.4224965706447188, 0.42158207590306357, 0.4336991312299954, 0.4357567443987197, 0.43621399176954734, 0.4444444444444444, 0.4444444444444444, 0.45038866026520347]
#plt.plot(range(1,21), accuracy_scores)
#plt.xlabel('Value of K for KNN')
#plt.ylabel('Testing Accuracy')
#plt.show()

# confusion matrix
from sklearn.metrics import plot_confusion_matrix
#class_names = ['soul and reggae', 'hip-hop', 'classical', 'jazz and blues', 'metal', 'dance and electronica', 'pop', 'folk', 'punk', 'classic pop and rock']
cmat = metrics.confusion_matrix(Ytest, Ypredict)
plt.figure(figsize=(6.5,6.5))
sn.heatmap(cmat, annot=True, annot_kws={"size": 13}, fmt='g')
plt.show()