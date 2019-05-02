# CNN_for_Dance_Music_Classification

## Executive Summary

**Problem Statement:** In the field of music information retrieval, convolutional neural networks (CNNs) have been successful at identifying different music genres by utilizing audio samples that are preprocessed into spectrogram images. I wanted to investigate how accurately a CNN could distinguish between more closely related styles of dance music, and more specifically between different styles of techno. 

**This project is split into two parts:**
1. Part I uses a CNN for a binary classification problem and explores how well a network can identify between two styles of dance music: techno and EDM. *Part I provides a more in-depth walk-through and explanation of all the data collection, preprocessing, and network design.* 

2. Part II uses a similar CNN for a multiclassification problem and explores how well the network can identify between five styles, or “genres” of techno. *Part II is more concise and utilizes all of the same processes from Part I.* 

**All code is in python and the tasks performed in this project include:**
- Acquiring data from Spotify’s API with the help of Spotipy library
- Preprocessing of mp3 files into spectrogram images using librosa library
- Binary Classification with a 1D and 2D CNN
- Multiclassification with a 1D and 2D CNN
