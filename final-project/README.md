# project-files

### match_genre_to_bow.py
takes in MSD genre tag file, the musiXmatch lyrics file, and the file to write to (filter3.txt) and outputs the songs genre tag and bag-of-words that have matching MSD song ids 
### filter3.txt: 
the output of match_genre_to_bow.py, includes the genre tags and the bag-of-words for each song to be reformatted for classification 
### filter_and_classify.py
takes in filter3.txt and reformats the genre tags and bag-of-words for each song to be input into the classifier,, runs the classifier 
### mxm_dataset_test.txt
### mxm_dataset_train.txt
datasets to combine for the bags-of-words/lyrics portion of each song 
### msd_genre_dataset.txt
dataset that encompassed the genre tags 
### vocab.txt 
list of the 5000 relevant words in the vocabulary for the bags-of-words 
### project-paper.pdf
