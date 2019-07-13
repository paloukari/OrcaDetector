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
from inference import create_network


def _save_audio_segments(stream_url,
                         stream_name,
                         segment_seconds,
                         iteration_seconds,
                         mix_with,
                         output_path,
                         verbose):
    """
    Uses ffmpeg (via CLI) to retrieve audio segments from audio_url
    and saves it to the local output_path.

    If mix_with exists, it mixes this file with the live feed source.
    The result 
    """

    file_name = f"{stream_name}_%02d.wav"
    output_file = os.path.join(output_path, file_name)
    mix_with = ''

    if os.path.exists(mix_with):
        mix_with = '-i {} -filter_complex amix=inputs=2:duration=first'.format(
            (mix_with))

    ffmpeg_cli = 'ffmpeg -y -i {} {} -t {} -f segment -segment_time {} {}'.format(
        (stream_url), (mix_with), (iteration_seconds), (segment_seconds), (output_file))

    if not verbose:
        ffmpeg_cli = ffmpeg_cli + ' -loglevel warning'
    os.system(ffmpeg_cli)


def _perform_inference(model, encoder, inference_samples_path):
    """
    Reads all *.wav audio segments in the specified folder, extracts the features
    performs inference and finally deletes the files. The files will be either 
    way overwritten by ffmpeg in the next iteration.

    To simulate an orca, drop some positive 1 second samples in the folder.
    """
    try:
        audio_segments = glob.glob(
            os.path.join(inference_samples_path, '*.wav'))
        end_of_segment = int(
            orca_params.FILE_SAMPLING_SIZE_SECONDS*orca_params.LIVE_FEED_SAMPLING_RATE)

        # The features extraction should not take more than 3 seconds for 3*10 1 second segments
        features = [[segment, extract_segment_features('{}:0:{}'.format(
            (segment), (end_of_segment)))] for segment in audio_segments]

        print('Performing inference for {} audio segments'.format(
            (len(audio_segments))))
        x = features[:, 1]
        
        results = model.predict(x=x,
                                batch_size=orca_params.BATCH_SIZE,
                                verbose=1)

        shutil.rmtree(inference_samples_path)
    except:
        print('Unable to perform inference for {}'.format(
            (inference_samples_path)))

@click.command(help="Performs inference on the specified OrcaSound Live Feed source(s).",
               epilog=orca_params.EPILOGUE)
@click.option('--model-name',
              help='Specify the model name to use.',
              default=orca_params.DEFAULT_MODEL_NAME,
              show_default=True,
              type=click.Choice(
                  choices=orca_params.MODEL_NAMES))
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
@click.option('--weights-path',
              help='Specify the weights path to use.',
              default=os.path.join(orca_params.OUTPUT_PATH,
                                   'orca_weights_latest.hdf5'),
              show_default=True)
@click.option('--verbose',
              help='Sets the ffmpeg logs verbosity.',
              show_default=True,
              default=False)
def live_feed_inference(model_name,
                        stream_name,
                        segment_seconds,
                        sleep_seconds,
                        iteration_seconds,
                        weights_path,
                        verbose,
                        live_feed_path=orca_params.LIVE_FEED_PATH,
                        positive_samples_path=orca_params.POSITIVE_SAMPLES_PATH):
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

    In each iteration, a single positive sample file is selected, if any, and mixed with the live sources
    to simulate a positive signal. The file is deleted after usage.
    """

    # Create the network first

    model, encoder = create_network(model_name, weights_path)

    counter = 0

    while True:
        positive_sample = ''
        positive_samples = glob.glob(
            os.path.join(positive_samples_path, '*.wav'))

        if len(positive_samples) > 0:
            positive_sample = positive_samples[0]

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

                recording_samples_path = os.path.join(
                    live_feed_path, str(counter % 2))
                inference_samples_path = os.path.join(
                    live_feed_path, str((counter+1) % 2))

                # make sure the folders exist
                if not os.path.exists(recording_samples_path):
                    os.mkdir(recording_samples_path)
                if not os.path.exists(inference_samples_path):
                    os.mkdir(inference_samples_path)

                thread = Thread(target=_save_audio_segments, args=(stream_url,
                                                                   _stream_name,
                                                                   segment_seconds,
                                                                   iteration_seconds,
                                                                   positive_sample,
                                                                   recording_samples_path,
                                                                   verbose, ))
                threads.append(thread)
                thread.start()
            except:
                print(f'Unable to load stream from {stream_url}')

        results = _perform_inference(model, encoder, inference_samples_path)
        print(results)

        _ = [t.join(orca_params.LIVE_FEED_ITERATION_SECONDS) for t in threads]

        if os.path.exists(positive_sample):
            os.remove(positive_sample)
            print(f'{positive_sample} deleted.')

        if sleep_seconds > 0:
            print(
                f'Sleeping for {sleep_seconds} seconds before starting next interation.\n')
            time.sleep(sleep_seconds)
