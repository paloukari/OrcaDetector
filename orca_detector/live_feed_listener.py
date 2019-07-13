# -*- coding: future_fstrings -*-

"""
Script to listen to a live audio feed. The audio can either be played in the system audio output, or saved to a file.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import m3u8
import numpy as np
import orca_params
import os
import random
import time
import uuid
import urllib.request

# Dictionary of stream base URLs; used in building stream links
stream_bases = {
    'OrcasoundLab': 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab',
    'BushPoint': 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_bush_point',
    'PortTownsend': 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_bush_point/hls/1562956219/live.m3u8'
}

def _play_audio(audio_url, output_path, segmentDurationSeconds):
    """
    Uses ffmpeg (via CLI) to retrieve an audio segment from audio_url
    and play it to the default audio output
    """
    print(f'Playing audio segment: {audio_url}')

    ffmpeg_cli = f'ffmpeg -y -i {audio_url} -t {segmentDurationSeconds} -f pulse "stream name" -loglevel warning'
    os.system(ffmpeg_cli)


def _save_audio(audio_url, output_path, segmentDurationSeconds):
    """
    Uses ffmpeg (via CLI) to retrieve an audio segment from audio_url
    and save it to the local output_path.
    """

    # generate random file names in case stream reuses its names
    file_name = f"{uuid.uuid4().hex}.wav"
    output_file = os.path.join(output_path, file_name)
    print(f'Saving audio segment: {audio_url}')
    print(f' to {output_file}')

    ffmpeg_cli = f'ffmpeg -y -i {audio_url} {output_file} -t {segmentDurationSeconds} -loglevel warning'
    os.system(ffmpeg_cli)


def collect(streamName, save=False, data_path=orca_params.DATA_PATH):
    """
    Connects to audio streams in the `streams` dictionary, selects a random segment
    to record, then sleeps for 1-15 minutes before repeating.  The loop never
    ends, so to exit, press CTRL-C or kill the process.
    """

    while True:
        for stream_name, stream_base in stream_bases.items():
            try:
                # get the ID of the latest stream and build URL to load
                latest = f'{stream_base}/latest.txt'
                stream_id = urllib.request.urlopen(
                    latest).read().decode("utf-8").replace('\n', '')
                stream_url = f'{stream_base}/hls/{stream_id}/live.m3u8'
                output_path = os.path.join(data_path, 'Noise/', stream_name)
                stream_obj = m3u8.load(stream_url)
                # pick a single audio segment from the stream
                if len(stream_obj.segments) > 0:
                    audio_segment = random.choice(stream_obj.segments)
                    base_path = audio_segment.base_uri
                    file_name = audio_segment.uri
                    audio_url = base_path + file_name
                    if save:
                        _save_audio(audio_url, output_path)
                    else:
                        _play_audio(audio_url, output_path)
            except:
                print(f'Unable to load stream from {stream_url}')

        # sleep for 1-15 minutes
        sleep_sec = np.random.randint(low=60, high=900)
        print(
            f'Sleeping for {sleep_sec} seconds before getting next sample.\n')
        time.sleep(sleep_sec)


if __name__ == '__main__':

    # parse command line parameters and flags
    parser = argparse.ArgumentParser(description='OrcaDetector - W251 (Summer 2019)',
                                     epilog='by Spyros Garyfallos, Ram Iyer, Mike Winton')

    parser.add_argument('--streamName',
                        type=str,
                        choices=['OrcasoundLab', 'BushPoint', 'PortTownsend'],
                        help='Specify the hydrophone live feed stream to listen to.')

    parser.add_argument('--save', action='store_true',
                        help='Save to files instead of playing.')

    parser.add_argument('--segmentDurationSeconds',
                        type=int,
                        help='Defines how many seconds ech iteration segment will be.')

    parser.add_argument('--sleepSeconds',
                        type=int,
                        help='Seconds to sleep between each iteration.')

    args = parser.parse_args()

    if not args.streamName:
        streamName = 'BushPoint'
    else:
        streamName = parser.streamName

    if not args.segmentDurationSeconds:
        segmentDurationSeconds = orca_params.LIVE_FEED_SLEEP_SEGMENT_SECONDS
    else:
        segmentDurationSeconds = parser.segmentDurationSeconds

    if not args.sleepSeconds:
        sleepSeconds = orca_params.LIVE_FEED_SLEEP_SECONDS
    else:
        sleepSeconds = parser.sleepSeconds

    if args.save:
        collect(streamName=streamName, segmentDurationSeconds=segmentDurationSeconds,
                sleepSeconds=sleepSeconds, save=True)
    else:
        collect(streamName=streamName,
                segmentDurationSeconds=segmentDurationSeconds,  sleepSeconds=sleepSeconds)
