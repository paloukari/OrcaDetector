# -*- coding: future_fstrings -*-

"""
Script to collect random samples of noise from live streams.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import m3u8
import numpy as np
import orca_params
import os
import random
import time
import uuid

# Dictionary of streams we're recording from
streams = {'OrcasoundLab': 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab/hls/1562344334/live.m3u8'}

def _save_audio(audio_url, output_path):
    """
    Uses ffmpeg (via CLI) to retrieve an audio segment from audio_url
    and save it to the local output_path.
    """

    # generate random file names in case stream reuses its names
    file_name = f"{uuid.uuid4().hex}.wav"
    output_file = os.path.join(output_path, file_name)
    print(f'Saving audio segment: {audio_url}')
    print(f' to {output_file}')
    
    ffmpeg_cli = f'ffmpeg -y -i {audio_url} {output_file} -loglevel warning'
    os.system(ffmpeg_cli)
  
    
def collect(data_path=orca_params.DATA_PATH):
    """
    Connects to audio streams in the `streams` dictionary, selects a random segment
    to record, then sleeps for 1-15 minutes before repeating.  The loop never
    ends, so to exit, press CTRL-C or kill the process.
    """
    
    while True:
        for stream_name, stream_url in streams.items():
            output_path = os.path.join(data_path, 'Noise/', stream_name)
            stream_obj = m3u8.load(stream_url)
            # pick a single audio segment from the stream
            audio_segment = random.choice(stream_obj.segments)
            base_path = audio_segment.base_uri
            file_name = audio_segment.uri
            audio_url = base_path + file_name
            _save_audio(audio_url, output_path)
            
        # sleep for 1-15 minutes
        sleep_sec = np.random.randint(low=60, high=900)
        print(f'Sleeping for {sleep_sec} seconds before getting next sample.\n')
        time.sleep(sleep_sec)

if __name__ == '__main__':
    collect()
