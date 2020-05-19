
import sys
import nltk

# file with genre tags, song title, artist name
genre_file = open(sys.argv[1], "r")
# file with lyrics
lyrics_file = open(sys.argv[2], "r")
# file to write to
filter_file = open(sys.argv[3], "w")


genre_lines = genre_file.readlines()
lyrics_lines = lyrics_file.readlines()


# matching the songIDs from the musiXmatch dataset to the MSD genre dataset and
# grabbing the bags-of-words and genre tag

### OUTPUT GOES TO FILTER3.txt 

for line in genre_lines:
    genre = line.split(",")[0]
    song_id1 = line.split(",")[1]
    artist = line.split(",")[2]
    song_name = line.split(",")[3]

    for lin in lyrics_lines:
        if not line.startswith("%"):
            song_id2 = lin.split(",")[0]
            if song_id1 == song_id2:
                filter_line = [genre, lin.split(",")[2:]]
                filter_file.write(str(filter_line).strip("[]"))

filter_file.close()
lyrics_file.close()
genre_file.close()
