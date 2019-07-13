# -*- coding: future_fstrings -*-

"""
Script to listen to a live audio feed. The audio can either be played in the system audio output, or saved to a file.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import click
import glob
import m3u8
import numpy as np
import orca_params
import os
import random
import shutil
import time
from threading import Thread
import uuid
import urllib.request
from database_parser import extract_segment_features


def _save_audio_segments(stream_url,
                         stream_name,
                         segment_seconds,
                         iteration_seconds,
                         output_path,
                         verbose):
    """
    Uses ffmpeg (via CLI) to retrieve audio segments from audio_url
    and saves it to the local output_path.
    """

    file_name = f"{stream_name}_%02d.wav"
    output_file = os.path.join(output_path, file_name)
    print(f'Saving audio segments: {stream_url}')
    print(f' to {output_file}')

    ffmpeg_cli = 'ffmpeg -y -i {} -t {} -f segment -segment_time {} {}'.format(
        (stream_url), (iteration_seconds), (segment_seconds), (output_file))

    if not verbose:
        ffmpeg_cli = ffmpeg_cli + ' -loglevel warning'
    os.system(ffmpeg_cli)


def _perform_inference(inference_output_path):
    """
    Reads all *.wav audio segments in the specified folder, extracts the features
    performs inference and finally deletes the files. The files will be either 
    way overwritten by ffmpeg in the next iteration.

    To simulate an orca, drop some positive 1 second samples in the folder.
    """
    try:
        audio_segments = glob.glob(os.path.join(inference_output_path, '*.wav'))
        end_of_segment = int(
            orca_params.FILE_SAMPLING_SIZE_SECONDS*orca_params.LIVE_FEED_SAMPLING_RATE)
        
        # The features extraction should not take more than 3 seconds for 3*10 1 second segments
        features = [[segment, extract_segment_features('{}:0:{}'.format(
            (segment), (end_of_segment)))] for segment in audio_segments]

        print(f'Performing inference for {len(audio_segments)} audio segments')

        # TODO: Perform the inteference here and measure the time duration.

        shutil.rmtree(inference_output_path)
    except:
        print(f'Unable to perform inference for {inference_output_path}')

@click.command(help="Performs inference on the specified OrcaSound Live Feed source(s).",
               epilog=orca_params.EPILOGUE)
@click.option('--stream-name',
              help='Specify the hydrophone live feed stream to listen to.',
              default='All',
              show_default=True,
              type=click.Choice(orca_params.ORCASOUND_STREAMS_NAMES))
@click.option('--segment-seconds',
              help='Defines how many seconds each audio segment will be.',
              show_default=True,
              default=orca_params.LIVE_FEED_SEGMENT_SECONDS)
@click.option('--sleep-seconds',
              help='Seconds to sleep between each iteration.',
              show_default=True,
              default=orca_params.LIVE_FEED_SLEEP_SECONDS)
@click.option('--iteration-seconds',
              help='Total seconds for each iteration.',
              show_default=True,
              default=orca_params.LIVE_FEED_ITERATION_SECONDS)
@click.option('--verbose',
              help='Sets the ffmpeg logs verbocity.',
              show_default=True,
              default=False)
def live_feed_inference(stream_name,
                        segment_seconds,
                        sleep_seconds,
                        iteration_seconds,
                        verbose,
                        live_feed_path=orca_params.LIVE_FEED):
    """
    Connects to specified audio stream(s) in the `ORCASOUND_STREAMS` dictionary, and records audio segments in iterations 
    and performs inference on the recorded data.

    IMPORTANT: The recording happens in two directories, active and passive. The inference is performed on the passive, 
    assuming that the inference is faster than the iteration length. Otherwise, there will time windows that wont be recorded.

    Each audio segment has length=segment_seconds, each iteration has length=iteration_seconds and between iterations,
    a break of sleep_seconds happens.
    The loop never ends, so to exit, press CTRL-C or kill the process.
    The iterations are required because the latest feed URI will change over time and needs to
    be recalcuated.
    """

    counter = 0
    while True:
        counter = counter + 1
        threads = []
        for _stream_name, _stream_base in orca_params.ORCASOUND_STREAMS.items():
            if stream_name != 'All' and stream_name != _stream_name:
                continue

            try:
                # get the ID of the latest stream and build URL to load
                latest = '{}/latest.txt'.format((_stream_base))
                stream_id = urllib.request.urlopen(
                    latest).read().decode("utf-8").replace('\n', '')
                stream_url = '{}/hls/{}/live.m3u8'.format(
                    (_stream_base), (stream_id))

                recording_output_path = os.path.join(
                    live_feed_path, str(counter % 2))
                inference_output_path = os.path.join(
                    live_feed_path, str((counter+1) % 2))

                # make sure the folders exist

                if not os.path.exists(recording_output_path):
                    os.mkdir(recording_output_path)
                if not os.path.exists(inference_output_path):
                    os.mkdir(inference_output_path)

                thread = Thread(target=_save_audio_segments, args=(stream_url, _stream_name,
                                                                   segment_seconds, iteration_seconds,
                                                                   recording_output_path, verbose, ))
                threads.append(thread)
                thread.start()
            except:
                print(f'Unable to load stream from {stream_url}')

        _perform_inference(inference_output_path)

        _ = [t.join(orca_params.LIVE_FEED_ITERATION_SECONDS) for t in threads]

        if sleep_seconds > 0:
            print(
                f'Sleeping for {sleep_seconds} seconds before starting next interation.\n')
            time.sleep(sleep_seconds)
