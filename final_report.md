# OrcaDetector Final Project Report

This UC Berkeley Master of Information in Data Science final course project was developed by
[Spyros Garyfallos](mailto:spiros.garifallos@berkeley.edu ), [Ram Iyer](mailto:ram.iyer@berkeley.edu), and [Mike Winton](mailto:mwinton@berkeley.edu) for the W251 "Deep Learning in the Cloud and at the Edge" course (Summer 2019 term).

## Abstract

This paper applies the previously published [VGGish audio classification model](TBD) to classify the species of marine mammals based on audio samples.  We use a distant learning approach, beginning with model weights that were pretrained on Google's published [Audioset](TBD) data.  We then finish training with a strongly supervised dataset from [Watkins Marine Mammal Sound Database](https://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm).  We achieve an overall F1 score of TBD over TBD species, with TBD of the species achieving an F1 score > 0.80.  We then deploy the trained model to an [NVIDIA Jetson TX2](TBD) edge computing device to perform inference locally, simulating a deployment connected to a hydrophone in the middle of the ocean without internet connectivity.   Since we don't have access to our own hydrophone, for the purposes of simulation, we connect to the [live.orcasound.net](http://live.orcasound.net) live audio stream and perform inference on this stream.  We also incorporate the ability for a person to "inject" an audio sample from a marine mammal species into the live audio stream to simulate an actual detection event.

## Introduction

- why are marine mammals interesting to scientists? 
- how would this model be useful to scientists?
- how is data collected, and why is the disconnected nature of a TX2-like device useful?

## Background

- how is audio classification generally done (audio -> image)?
- what is a mel spectrogram
- give a few different examples from the links we collected
- what kind of accuracy has been demonstrated in similar use cases?

## Current Contributions

- collecting Noise data from multiple live streams (i.e. mammal not detected)
- applying VGGish to marine mammal audio samples for supervised ML
- demonstrating the trained model can be used for inference on a TX2
- demonstration of noise + positive sample mixing technique, used for simulation now, but which could be used in the future to augment a training dataset

## Model Architecture

- describe VGGish and its differences from VGG
- mention of logistic regression as a baseline

## Method

- short intro paragraph
- mention machine types (train on K80/V100; inference on TX2)?

### Dataset

- crawling the Watkins data (with permission)
- quantizing and converting to spectrograms
- stratified train/val/test split
- selected EDA plots
- choice of species what were included vs. excluded (based on too few training samples)
- collection of Noise data from live streams

### Evaluation Metrics

- F1 and overall accuracy?

### Experimental Results

- baseline with logistic reg; mostly with VGGish
- tabular summary of subset of hyperparameter optimization runs
- TensorBoard plots from the "final" training runs (MW: I have these)
- Summary of classification metrics for "final" run
- Discussion of which species performed well vs. poorly

## Live Stream Inference & Simulation

- describe architecture of handling live stream
- injection, ffmpeg mixing
- how Flash web app (or Jupyter notebook) works

## Conclusion

- impressive results, with model that easily fits on a TX2
- demonstrated that VGGish works well for marine mammal classification
- demonstrated that distant learning w/ Audioset YouTube samples (millions) followed by strongly supervised topical samples (thousands of audio files)
- room for future improvements depending on which species are most interesting

## Future Work

- generate augmented training dataset by using our simulation technique to generate "blurred" audio files by mixing different levels of noice with the mammal sounds.
- more work on the few poorly classified species
- run the live stream inference for a few weeks to see if we detect any species organically

## References

 
