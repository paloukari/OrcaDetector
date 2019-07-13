# -*- coding: future_fstrings -*-

"""
Script to listen to a live audio feed. The audio can either be played in the system audio output, or saved to a file.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""
import argparse
import m3u8
import numpy as np
import orca_params
import os
import random
import time
from threading import Thread
import uuid
import urllib.request

# Dictionary of stream base URLs; used in building stream links
stream_bases = {
    'OrcasoundLab': 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab',
    'BushPoint': 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_bush_point',
    'PortTownsend': 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_port_townsend'
}


def _play_audio(audio_url, iteration_seconds):
    """
    Uses ffmpeg (via CLI) to retrieve an audio segment from audio_url
    and play it to the default audio output device for the 
    specified duration.
    """
    print(f'Playing audio segment: {audio_url}')

    ffmpeg_cli = f'''ffmpeg -y -i {audio_url} -t 
        {iteration_seconds} -f pulse "stream name" -loglevel warning'''
    os.system(ffmpeg_cli)


def _save_audio_segments(stream_url, stream_name, segment_seconds, iteration_seconds, output_path):
    """
    Uses ffmpeg (via CLI) to retrieve audio segments from audio_url
    and saves it to the local output_path.
    """

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    file_name = f"{stream_name}_%02d.wav"
    output_file = os.path.join(output_path, file_name)
    print(f'Saving audio segments: {stream_url}')
    print(f' to {output_file}')

    ffmpeg_cli = f'ffmpeg -y -i {stream_url} -t {iteration_seconds} -f segment -segment_time {segment_seconds} {output_file}' #-loglevel warning'
    os.system(ffmpeg_cli)


def record(stream_name, segment_seconds, sleep_seconds, iteration_seconds, live_feed_path=orca_params.LIVE_FEED):
    """
    Connects to specified audio stream(s) in the `stream_bases` dictionary, and records audio segments in iterations.
    Each segment has length=segment_seconds, each iteration has length=iteration_seconds and between iterations, 
    a break of sleep_seconds happens.
    The loop never ends, so to exit, press CTRL-C or kill the process.
    The iterations are required because the latest feed URI will change over time and needs to 
    be recalcuated.
    """

    while True:
        threads = []
        for _stream_name, _stream_base in stream_bases.items():
            if stream_name != 'All' and stream_name != _stream_name:
                continue

            try:
                # get the ID of the latest stream and build URL to load
                latest = f'{_stream_base}/latest.txt'
                stream_id = urllib.request.urlopen(
                    latest).read().decode("utf-8").replace('\n', '')
                stream_url = '{}/hls/{}/live.m3u8'.format(
                    (_stream_base), (stream_id))
                # os.path.join(live_feed_path, stream_name)
                output_path = live_feed_path

                thread = Thread(target=_save_audio_segments, args=(stream_url, _stream_name,
                                                              segment_seconds, iteration_seconds,
                                                              output_path,))
                threads.append(thread)
                thread.start()
            except:
                print(f'Unable to load stream from {stream_url}')

            

        _ = [t.join() for t in threads]

        if sleep_seconds > 0:
            print(
                f'Sleeping for {sleep_seconds} seconds before starting next interation.\n')
            time.sleep(sleep_seconds)


if __name__ == '__main__':

    # parse command line parameters and flags
    parser = argparse.ArgumentParser(description='OrcaDetector - W251 (Summer 2019)',
                                     epilog='by Spyros Garyfallos, Ram Iyer, Mike Winton')

    parser.add_argument('--stream_name',
                        type=str,
                        choices=['OrcasoundLab', 'BushPoint', 'PortTownsend', 'All'],
                        help='Specify the hydrophone live feed stream to listen to.')

    parser.add_argument('--segment_seconds',
                        type=int,
                        help='Defines how many seconds each audio segment will be.')

    parser.add_argument('--sleep_seconds',
                        type=int,
                        help='Seconds to sleep between each iteration.')

    parser.add_argument('--iteration_seconds',
                        type=int,
                        help='Total seconds for each iteration.')

    args = parser.parse_args()

    if not args.stream_name:
        stream_name = 'All'
    else:
        stream_name = parser.stream_name

    if not args.segment_seconds:
        segment_seconds = orca_params.LIVE_FEED_SEGMENT_SECONDS
    else:
        segment_seconds = parser.segment_seconds

    if not args.sleep_seconds:
        sleep_seconds = orca_params.LIVE_FEED_SLEEP_SECONDS
    else:
        sleep_seconds = parser.sleep_seconds

    if not args.iteration_seconds:
        iteration_seconds = orca_params.LIVE_FEED_ITERATION_SECONDS
    else:
        iteration_seconds = parser.iteration_seconds

    record(stream_name=stream_name,
           segment_seconds=segment_seconds,
           sleep_seconds=sleep_seconds,
           iteration_seconds=iteration_seconds)
