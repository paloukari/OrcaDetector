
"""
Functions to aid with the demo notebook

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import os
import shutil

import pandas as pd
import librosa
import librosa.display
import urllib

import matplotlib.pyplot as plt
from scipy.io import wavfile

import database_parser
import orca_params
import mel_params

def create_tmpfile(dir_name="./tmp/",file_name=None):
    """
    Creates a specified temp directory if it does not exist
    If the temp directory exists then all files in it are removed.
    If a file is specified, its contents are copied into the empty directory
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        file_names=os.listdir(dir_name)
        for file in file_names:
            os.remove(dir_name + file)
    if (file_name != None):
        print("Copying {} to {}".format(file_name,dir_name))
        shutil.copy(full_name,dir_name)

def get_noise_sample(stream_name,volume):
    stream_base = orca_params.ORCASOUND_STREAMS[stream_name]
    latest = '{}/latest.txt'.format(stream_base)
    stream_id = urllib.request.urlopen(
                    latest).read().decode("utf-8").replace('\n', '')
    stream_url = '{}/hls/{}/live.m3u8'.format(
                    (stream_base), (stream_id))

    create_tmpfile(dir_name="./noise_sounds/")
    file_name = "{}.wav".format(stream_name)
    cmd = 'ffmpeg -i {} -t 10 -filter:a "volume={}" ./noise_sounds/{}'.format(stream_url,volume,file_name)
    #print(cmd)
    os.system(cmd)
    file_name = "./noise_sounds/{}".format(file_name)
    return file_name

def get_combined_sample(mammal_name, mammal_volume,
                         noise_stream_name,noise_stream_volume):
    
    create_tmpfile(dir_name="./combined_sounds/")
    create_tmpfile(dir_name="./inference_output/")
    outfile_name = "./combined_sounds/output.wav"
    inference_file_name = "./inference_output/{}_%02d.wav".format(noise_stream_name)

    stream_base = orca_params.ORCASOUND_STREAMS[noise_stream_name]
    latest = '{}/latest.txt'.format(stream_base)
    stream_id = urllib.request.urlopen(
                    latest).read().decode("utf-8").replace('\n', '')
    stream_url = '{}/hls/{}/live.m3u8'.format(
                    (stream_base), (stream_id))
    
    filter_cmd = '[0:0]volume={}[a];[1:0]volume={}[b];[a][b]amix=inputs=2:duration=first'.format(noise_stream_volume, 
                                                                                             mammal_volume)

    mix_with_command = '-i {} -filter_complex "{}"'.format(mammal_name,filter_cmd)

    ffmpeg_cli_1 = 'ffmpeg -y -i {} {} -t 10  {}'.format(stream_url, mix_with_command,outfile_name)
    ffmpeg_cli_2 = 'ffmpeg -y -i {} {} -t 10 -f segment -segment_time 1 {}'.format(stream_url,
                                                                                   mix_with_command,
                                                                                   inference_file_name)
     
    #print (ffmpeg_cli_1)
    #print(ffmpeg_cli_2)
    os.system(ffmpeg_cli_1)
    os.system(ffmpeg_cli_2)
    
    return outfile_name

class MelUtils ():
    def display_mel(self,file):
        #print("In display mel for {}".format(file))
        # Load sound file
        y, sr = librosa.load(file)

        sr = mel_params.SAMPLE_RATE
        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=mel_params.NUM_BANDS)

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.core.amplitude_to_db(S)

        #print("Plot figure")
        # Make a new figure
        plt.figure(figsize=(12,4))
        
        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

        # Put a descriptive title on the plot
        plt.title('mel power spectrogram')

        # draw a color bar
        plt.colorbar(format='%+02.0f dB')

        # Make the figure layout compact
        plt.tight_layout()
        plt.show()
        
    def display_wave(self,file):
        #print("In display wave")
        y, sr = librosa.load(file)
        plt.figure(figsize=(12,4))
        librosa.display.waveplot(y,sr=mel_params.SAMPLE_RATE)
        plt.title('Audio Waveform')
        plt.show()


class MammalFind(object):

    def __init__(self):
        """
            Build a dictionary of labels to filenames
            Sort by labels and per label by audio duration
        """
        self.root_dir = orca_params.DATA_PATH
        all_samples = database_parser.label_files(self.root_dir)
        dataset_flattened = [[label,file] for label in all_samples.keys() for file in all_samples[label]]
        self.file_df = pd.DataFrame(dataset_flattened) 
        self.file_df.columns = ['label','fname']
        #print(self.file_df.groupby('label').count())
        self.file_df['duration'] = self.file_df['fname'].apply(self._extract_duration)
        self.file_df.sort_values(by=['label','duration'], ascending=[True,False], inplace=True)
        #print(self.file_df.head())
        if os.path.exists(orca_params.POSITIVE_INPUT_PATH) == False:
            os.mkdir(orca_params.POSITIVE_INPUT_PATH)

    def _extract_duration(self,fname):
        """
          helper function to get the duration per file
        """
        try:
            rate, data = wavfile.read(fname)
            duration = data.shape[0]/rate
        except Exception as e:
            print("Count not extract {} due to {}".format(fname,str(e)))
            duration = 0
        return duration
    
    def get_valid_labels(self):
        """
        remove labels that do not qualify for training as per orca_params.REMOVE_CLASSES
        """

        all_classes = set(self.file_df['label'].unique())
        remove_classes = set(orca_params.REMOVE_CLASSES)
        return list(all_classes - remove_classes)
    
    def get_sample_sound(self, fname,volume,play_time=10):
        out_file_name = fname
        if (abs(volume - 0.0001) == 1):
            ipd.display(ipd.Audio(fname))
        else:
            #use ffmpeg to create modulated file
            out_file_name = "./display_sounds/output.wav" 
            create_tmpfile(dir_name="./display_sounds/")
            cmd = 'ffmpeg -ss 0 -t {} -i {} -filter:a "volume={}" ./display_sounds/output.wav'.format(play_time,fname,volume)
            #print(cmd)
            os.system(cmd)
            #print(cmd)
        return (out_file_name)
        
    def get_sample (self, mammal,verbose=False):
        """
        Get the longest duration sample for each mammal
        Initially implemented a random sample but then switched to longest duration sample
            to see if the success rate is better
        """
        fnames = self.file_df[self.file_df.label == mammal]
        fnames = fnames[(fnames.duration > 5) & (fnames.duration < 15)  ]
        if (fnames.shape[0] == 0):
            fnames = fnames.iloc[0]
        #fnames = fnames.sample(1)
        if (verbose):
            print(fnames.iloc[0:min(10,fnames.shape[0])])
        fname = fnames.iloc[0]['fname']
        full_name = fname.replace("'s","\\'s")
        return full_name
