# CNN Binary Classification of Techno vs. EDM genres 

## Project Summary

### 1. Introduction
### 2. Notebook Structure 
Notebooks:
- __[1.1_EDM_Download_Tracks_and_Audio_Features.ipynb](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/1.1_EDM_Download_Tracks_and_Audio_Features.ipynb)__ EDM song data collection from Spotify and EDA of features
- __[1.2_Techno_Download_Tracks_and_Audio_Features.ipynb](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/1.2_Techno_Download_Tracks_and_Audio_Features.ipynb)__ Techno song data collection from Spotify and EDA of features
- __[1.3_Audio_File_Manipulation_and_Neural_Network.ipynb](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/1.3_Audio_File_Manipulation_and_Neural_Network.ipynb)__ Spectrogram processing with librosa and CNN Application

The file structure for the datasets used to complete this project are shown below. These additional folders and associated files can be downloaded here (with the exception of song_specs.npy and genres.npy, due to size constraints).

```
Part_I
  |
  |---datasets
  |       techno.csv      (dataframe of song ids, mp3 links, and audio features)
  |       edm.csv
  |       song_specs.npy  (processed mel-spec input array for network. Not Available for download)
  |       genres.npy      (associated label inputs for network. Not avaialble for download)
  |
  |---downloads
         |---techno       (folder containing 878 mp3 files)
         |---edm          (folder containing 929 mp3 files)
```

### 3. Data Collection 
**Data Selection:** Two publicly available Spotify playlists were chosen by performing a google search for “Techno” and “EDM” playlists. Each contained around 1500 songs and were representative of a unique genre of dance music. 

[image]

**Data Abstraction:** With the help of the python wrapper Spotipy, two different Spotify API endpoints were accessed to get a variety of different information: 
- __[Get A Playlist's Tracks](https://developer.spotify.com/documentation/web-api/reference/playlists/get-playlists-tracks/)__ was used to obtain a 30 second mp3 clip for all the songs in a playlist
- __[Get Audio Features for Several Tracks](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/)__ was used to collect additional song data from 12 categories such as tempo, danceability, etc. 

[data dictionary]

### 4. EDA 
The distribution of audio features were compared for the two playlists. For all 12 categories, the most significant differences were found to be the tempo and instrumentalness, which predicts if a song has vocals or not (0 = no vocals, 1 = has vocals) Not surprisingly, the average tempo for a techno song was 124 bpm while EDM was faster at 127 bpm, and the techno playlist was predicted to have almost no vocals while EDM had an inverse distribution with almost all vocals. 

The figure below displays the audio feature distributions for the techno playlist.

[image]

### 5. Spectrogram Conversion & Preprocessing
The mp3 files were converted to spectrogram images using the librosa library. There are many options for spectrogram conversion, which is discussed in more detail in Part I. Different conversions for a 30 second clip are shown below: Figure 1 displays the mp3 file as a one-dimensional pressure-time plot. A fourier transformation yields the mel-spectrogram, displayed in both Figures 2 and 3 with the only difference being the scale of loudness (in decibels) at the associated frequencies. 

[image ]


Before inputting into a network, the images were visually examined to identify if a human could distinguish between both genres. If we look at the melspectrograms for the first 30 tracks in each playlist (rotated so time is on the y-axis), we can see that there is a noticable difference between techno and EDM. A network should easily be able to learn these differences. 

[image ] 


### 6. Neural Network Architecture

[image ] 

### 7. Results
