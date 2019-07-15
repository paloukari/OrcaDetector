# -*- coding: future_fstrings -*-

"""
Script to collect random samples of noise from live streams.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import click
import m3u8
import numpy as np
import orca_params
import os
import random
import time
import uuid
import urllib.request


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
  
    

@click.command(help="Periodically records samples from the predefined OrcaSound Live Feed sources.",
               epilog=orca_params.EPILOGUE)
def collect_noise(data_path=orca_params.DATA_PATH):
    """
    Connects to audio streams in the `streams` dictionary, selects a random segment
    to record, then sleeps for 1-15 minutes before repeating.  The loop never
    ends, so to exit, press CTRL-C or kill the process.
    """
    
    while True:

        for stream_name, stream_base in orca_params.ORCASOUND_STREAMS.items():
            try:
                # get the ID of the latest stream and build URL to load
                latest = f'{stream_base}/latest.txt'
                stream_id = urllib.request.urlopen(latest).read().decode("utf-8").replace('\n','') 
                stream_url = f'{stream_base}/hls/{stream_id}/live.m3u8'
                output_path = os.path.join(data_path, 'Noise/', stream_name)
                stream_obj = m3u8.load(stream_url)
                # pick a single audio segment from the stream
                if len(stream_obj.segments) > 0:
                    audio_segment = random.choice(stream_obj.segments)
                    base_path = audio_segment.base_uri
                    file_name = audio_segment.uri
                    audio_url = base_path + file_name
                    _save_audio(audio_url, output_path)
            except:
                print(f'Unable to load stream from {stream_url}')
            
        # sleep for 1-15 minutes
        sleep_sec = np.random.randint(low=60, high=900)
        print(f'Sleeping for {sleep_sec} seconds before getting next sample.\n')
        time.sleep(sleep_sec)