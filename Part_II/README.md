# CNN Multiclassification of Five Techno Labels

## Project Summary
This report explains how I constructed a convolutional neural network to classify five different "genres" of techno music. Part_II executes all of the same actions performed in Part_I, but is much more concise without explanation of individual steps.

Tasks performed in this repo include:
- Acquiring mp3 data from Spotify’s API with the help of Spotipy library
- Preprocessing of mp3 files into mel-spectrogram images using librosa library
- Multiclassification of genres with a 1D CNN

### 1. Background
**Inspiration** In an effort to understand underground techno music more thouroughly, I was curious if a neural network could differentiate sub-genres of techno, or at least shed light on differences between many of the artists I listen to. Would you describe a song as minimal, or lo-fi, or ambient... this list goes on! If you could build a network to differentiate styles, then maybe I could use that network to analyze a 1.5 hour long mix and output what styles are detected. It might be fun to analyze how dynamic or uniform a dj set is, or use the network output to determine if I should listen to the mix or skip it. However, creating the required dataset would be problematic because techno is inherently about the evolution and creation of new sounds, sub-genres of techno are constantly evolving, and most techno tracks overlap many "genre" styles. Therefore, constructing a data set of songs with specific genre tags would be very daunting, and questionably subjective.

Instead, I decided to substitute the record label as a genre (which is how electronic music styles are typically categorized and recognized) and investigate if a network could distinguish between labels.

### 2. Notebook Structure
- All tasks are performed in __[notebook 2.1](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_II/2.1_1D_CNN_for_Multiclassification_of_Techno_Labels.ipynb)__


- The file structure for the datasets used to complete this project are shown below. These additional folders and associated files can be downloaded from dropbox __[here](https://www.dropbox.com/sh/1oqs7l54u5pzpxj/AADCodQE2mG1pfFszu3t8V7Ca?dl=0)__ (with the exception of song_specs.npy and genres.npy, due to size constraints).

```
Part_I
  |
  |---datasets
  |       dirtybird.csv   (dataframe of song ids, mp3 links, and audio features)
  |       drumcode.csv
  |       kompakt.csv
  |       lobster_theramin.csv
  |       ostgut_ton.csv
  |       song_specs.npy  (processed mel-spec input array for network. Not available for download)
  |       genres.npy      (associated label inputs for network. Not available for download)
  |
  |---downloads
         |---dirtybird          (folder containing 878 mp3 files)
         |---drumcode           (folder containing 929 mp3 files)
         |---kompakt            (folder containing 929 mp3 files)
         |---lobster_theramin   (folder containing 929 mp3 files)
         |---ostgut_ton         (folder containing 929 mp3 files)
```
### 3. Data Collection
**Data Selection:**
Five publicly available Spotify playlists were chosen by performing a google search for the respective record label name. Each label represents a different "genre", or fairly recognizable style of techno music. The number of usable tracks in each playlist ranged from 145 to 439 tracks. It was interesting to find that only 25% of tracks in the Lobster Theramin playlist had an available mp3 link. The reasons for this might be due to different licensing restrictions from this label, or perhaps using a different access token (such as Authorization Code flow rather than Client Credentials) would provide the link.

|Record label| (My Personal) Description of Label | Spotify link|# of Tracks in Playlist| # of Tracks with mp3|
|---| --- | --- |--- | --- |
| Drumcode |percussive, festival techno, filter sweeps common |https://open.spotify.com/playlist/2a9vewgAKwZoYkJVBqJoLH | 712 | 322 |
| Ostgut Ton |dark, experimental, minimal, heady |https://open.spotify.com/playlist/6Edpq3cmRdPdIyzU4J80TC | 318 | 318 |
| Lobster Theramin | lofi, emotive, acid, rolling textures| https://open.spotify.com/playlist/215TGFgN1aCZ94BBouUYKv | 568 | 145 |
| Kompakt | minimal, ambient, experimental|https://open.spotify.com/playlist/7nU5hYoDxu0DmdRm2DQRUt, https://open.spotify.com/playlist/3kCDl9f7jQ8sjVN8wInerl |445 | 439 |
| Dirtybird | deep house / tech house, club tracks|https://open.spotify.com/playlist/2XlbQn0hmv6eH8C16CGIN2 | 244| 205|






### 4. Spectrogram conversion and preprocessing
See __[Part_I](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/README.md)__ for explanation.

### 5. Neural Network Architecture
The same network structure optimized in Part_I was applied here without further optimization. See __[Part_I](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/README.md)__ for explanation.

![Alt text](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/images/network_architecture.png)

### 6. Results
The 1D CNN was able to predict the correct techno label with 66% accuracy on the training set and 56% accuracy on the test set. Interestingly, a network consisting of only one convolution layer produced similar results, so the network did not appear to be learning more with additional layers. Compared to Part_I, this model could likely have been improved with a different dimension reduction layer other than global max pooling, which is probably recognizing if a certain musical sound is present anywhere in the sound clip and could be contributing to missclassification.

![Alt text](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_II/images/results.png)

Also not surprisingly, the easiest label to recognize was Drumcode (which has a very distinct flavor), while the hardest was Lobster Theramin, which is certainly more diverse, but also had the least number of samples available.

|Label| Total # in Test Set| Overall Accuracy| Most likely to be Misclassified As|
|---|---|---|---|
|Kompakt| 440| 53% | Drumcode > Ostgut Ton |
|Drumcode|320| 82.2 % | Kompakt or Dirtybird |
|Dirtybird|210| 55.2% | Drumcode |
|Lobster Theramin| 140| 20% | Drumcode or Dirtybird |
|Ostgut ton| 320| 49.9% | Kompakt |
