# CNN Binary Classification of Techno vs. EDM genres 

## Project Summary
This report explains how I constructed a convolutional neural network to classify two different genres of dance music: techno and EDM. Tasks performed in this repo include:
- Acquiring data from Spotify’s API with the help of Spotipy library
- Preprocessing of mp3 files into spectrogram images using librosa library
- Binary Classification with a 1D and 2D CNN
### 1. Background
**Inspiration** I am a dance music enthusiast! My inspiration for this project came from my love and strong dislike of certain styles of dance music, and my desire to understand why those differences exist. I wanted to see just how well a network could distinguish two genres that are arguably very similar, yet different enough that only one draws me to the dance floor.  *Disclaimer: while music genre tags are a necessary evil and will always be helpful, genre distinctions can be restrictive, can't always be trusted, can be rejected by artists themselves, and are always evolving. Genres help us organize the world, but they are not bulletproof.* 

**What do I mean by techno and EDM?**
The words "Techno" and "EDM" both have ambiguous definitions, dependent on the individual and context. Each could be referring to a more specific style, OR intended to mean a broader, umbrella term that includes many unique styles. 
- Any type of dance music made with computers is technically electronic dance music, but there is also a specific style of dance music commonly referred to as EDM (short for electronic dance music), which is the popular, commercial style heard at American music festivals by artists such as Diplo, Calvin Harris, Tiesto, etc. 
- Techno is a style of dance music that originated in Detroit in the late 1980s and has since evolved into many sub-genres. Someone using the term techno could be referring to the more specific, original Detroit style, or one of the many sub-styles. 


### 2. Notebook Structure 
Notebooks:
- __[1.1_EDM_Download_Tracks_and_Audio_Features.ipynb](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/1.1_EDM_Download_Tracks_and_Audio_Features.ipynb)__ EDM song data collection from Spotify and EDA of features
- __[1.2_Techno_Download_Tracks_and_Audio_Features.ipynb](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/1.2_Techno_Download_Tracks_and_Audio_Features.ipynb)__ Techno song data collection from Spotify and EDA of features
- __[1.3_Audio_File_Manipulation_and_Neural_Network.ipynb](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/1.3_Audio_File_Manipulation_and_Neural_Network.ipynb)__ Spectrogram processing with librosa and CNN Application

The file structure for the datasets used to complete this project are shown below. These additional folders and associated files can be downloaded from dropbox __[here](https://www.dropbox.com/sh/06njtqz884z1yll/AAAvbDWOxhGIBsyO9YWlJqGua?dl=0
)__ (with the exception of song_specs.npy and genres.npy, due to size constraints).

```
Part_I
  |
  |---datasets
  |       techno.csv      (dataframe of song ids, mp3 links, and audio features)
  |       edm.csv
  |       song_specs.npy  (processed mel-spec input array for network. Not available for download)
  |       genres.npy      (associated label inputs for network. Not available for download)
  |
  |---downloads
         |---techno       (folder containing 878 mp3 files)
         |---edm          (folder containing 929 mp3 files)
```

### 3. Data Collection 
**Data Selection:** Two publicly available Spotify playlists were chosen by performing a google search for “Techno” and “EDM” playlists. Each contained around 1500 songs and were representative of a unique genre of dance music. 

![Alt text](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/images/playlist_header.png)

**Data Abstraction:** With the help of the python wrapper __[Spotipy](https://spotipy.readthedocs.io/en/latest/)__, two different Spotify API endpoints were accessed to get a variety of different information: 
- __[Get A Playlist's Tracks](https://developer.spotify.com/documentation/web-api/reference/playlists/get-playlists-tracks/)__ was used to obtain a 30 second mp3 clip for all the songs in a playlist. The number of useable tracks in the dataset was reduced by ~50% because only half of each playlists' tracks contained an mp3 link. 
- __[Get Audio Features for Several Tracks](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/)__ was used to collect additional song data from 12 categories such as tempo, danceability, etc. 


#### Data Dictionary for Spotify Track Information from Different Endpoints
*1  = Get a Playlist's Tracks, 2 = Get Audio Features for Several Tracks*

|API Endpoint| Feature | Description |
|---| ---|---|
|1| track_id | unique id given to every track|
|1| mp3 | downloadable link to 30 seconds of audio |
|1| track_name | |
|1| artist_name | |
|1| album_date | |
|1| track_length | |
|1| track_popularity | |
|2| acousticness | A confidence measure from 0.0 to 1.0 of whether the track is acoustic (1.0 = acoustic)|
|2| danceability | How suitable a track is for dancing from 0.0 to 1.0 |
|2| duration_ms |The duration of the track in milliseconds |
|2| energy |Measure of intensity of activity from 0.0 to 1.0 |
|2| instrumentalness | Predicts whether a track contains no vocals. 0 = has vocals, 1.0 = no vocal content|
|2| key | The key the track is in. Integers map to pitches using standard Pitch Class notation |
|2| loudness |The overall loudness of a track in decibels  |
|2| mode | Modality of track. 0 = minor, 1 = minor|
|2| speech | Detects presence of spoken words from 0 to 1.0 (values above 0.66 likely all spoken word) |
|2| tempo | Overall estimated tempo of a track in BPM. |
|2| time_signature | An estimated overall time signature for how many beats in a measure |
|2| valence | Musical positiveness measured from 0.0 to 1.0 (sad/negative to happy/positive)|


### 4. EDA 
The distribution of audio features were compared for the two playlists. For all 12 categories, the most significant differences were found to be with tempo and instrumentalness, which predicts if a song has vocals or not (0 = no vocals, 1 = has vocals) Not surprisingly, the average tempo for a techno song was 124 bpm while EDM was faster at 127 bpm, and the techno playlist was predicted to have almost no vocals while EDM had an inverse distribution with almost all vocals. 


![Alt text](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/images/histograms.png)


### 5. Spectrogram Conversion & Preprocessing
The mp3 files were converted to mel-spectrogram images using the librosa library. There are many options for spectrogram conversion, which is discussed in more detail in __[Notebook 1.3](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/1.3_Audio_File_Manipulation_and_Neural_Network.ipynb)__. Different conversions for a 30 second clip are shown below: Figure 1 displays the mp3 file as a one-dimensional pressure-time plot. A fourier transformation yields the mel-spectrogram, displayed in both Figures 2 and 3 with the only difference being the scale of loudness (in decibels) at the associated frequencies. 

![Alt text](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/images/spectrograms.png)


Before inputting into a network, the images were visually examined to identify if a human could distinguish between both genres. If we look at the mel-spectrograms for the first 30 tracks in each playlist (rotated so time is on the y-axis), we can see that there is a noticeable difference between techno and EDM. A network should easily be able to learn these differences. 

![Alt text](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/images/30_melspecs.png)


### 6. Neural Network Architecture
The inspiration for the network architecture came from a deep learning project __[Sander Dieleman published while interning at Spotify](http://benanne.github.io/2014/08/05/spotify-cnns.html)__
- **Input:** To create more samples for the network each 30 second track was divided into ten, resulting in an input of 3 seconds, or 128 time frames by 128 frequency bins. The input image was rotated for one-dimensional convolutions to occur on the time axis only.  
- **Network:** This network consists of 3 convolutional layers, all with ReLU activation functions. Each convolution is followed by batch normalization to speed up learning (by normalizing values and reducing oscillations in gradient descent for faster convergence), and max pooling to reduce the spatial size of the output for fewer parameters. Next, a global max pooling layer reduces the convolution dimensions by outputing a single max value for every feature map. The output can then be fed into the remaining dense layers, with the final output predicting two classes of music genres. 
- **Training** The network was implemented in Keras and trained to minimize accuracy of the predictions. Depending on the parameters chosen, the one-dimensional network required 5-11 minutes to train on my personal laptop. Much time was spent investigating how factors such as dropout, regularization, kernel size, batch size, etc., affected the performance and the shape of the loss/accuracy curves. A similar architecture was tested in a 2D network, but required > 1 hr and was not further tested.



![Alt text](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/images/network_architecture.png)

### 7. Results
The most favorable result was selected as the one with the most ideal loss/accuracy curves as well as the final accuracy scores (without being overfit). The final 1D CNN was able to predict the genre with 94% accuracy on the training set and 93% accuracy on the test set. 
![Alt text](https://github.com/amytaylor330/CNN_for_Dance_Music_Classification_repost/blob/master/Part_I/images/1D_results.png)

The most important factors affecting the model's performance were found to be:
- Shape of spectrogram: time on y-axis (for direction of convolution), and normalizing the values (results in a smoother learning rate)
- Dropout placement and amount: didn't do much when placed after a convolution, but necessary in the dense layers to reduce overfitting.
- Minibatch gradient descent: reduced overfitting (good results with batch size around 3000-5000)

Less important but still crucial
- Regularization: to reduce overfitting and control the rate of learning 
- Number of epochs: ideal number changed for different parameters, but was reliable at around 20 epochs. 

Parameters that barely changed final result:
- Kernel size
- Max pool stride: increasing the stride greatly increases the training time

**Future work** With unlimited time and GPU I would spent more time investigating topics such as:
- Substituting the global max pool layer with alternatives, such as global average pooling or LSTM
- Different activation functions (I only tried ReLU)
- Analyzing the feature maps and learned features
