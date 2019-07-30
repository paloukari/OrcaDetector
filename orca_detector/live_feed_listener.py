# -*- coding: future_fstrings -*-

"""
Script to listen to a live audio feed. The audio can either be played in the system audio output, or saved to a file.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import click
import datetime
import glob
import m3u8
import numpy as np
import orca_params
import os
import pandas as pd 
import random
import shutil
import sys
import time
import uuid
import urllib.request
from threading import Thread
from database_parser import extract_segment_features
from inference import create_network

POSITIVE_INFERENCE_TIMESTAMP = datetime.datetime.now().isoformat('-')


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

    mix_with_command = ''
    if os.path.exists(mix_with):
        print(f'Mixing with {mix_with}')
        mix_with_command = f'-i {mix_with} -filter_complex amix=inputs=2:duration=first'

    ffmpeg_cli = f'ffmpeg -y -i {stream_url} {mix_with_command} -t {iteration_seconds} -f segment -segment_time {segment_seconds} {output_file}'

    if not verbose:
        ffmpeg_cli = ffmpeg_cli + ' -loglevel error'
    os.system(ffmpeg_cli)


def _perform_inference(model, encoder, inference_samples_path, probability_threshold):
    """
    Reads all *.wav audio segments in the specified folder, extracts the features
    performs inference and finally deletes the files. The files will be either 
    way overwritten by ffmpeg in the next iteration.

    To simulate an orca, drop some positive 1 second samples in the folder.
    """

    results = []
    try:
        audio_segments = glob.glob(
            os.path.join(inference_samples_path, '*.wav'))
        if len(audio_segments) == 0:
            print('No audio segments found in {}'.format(
                (inference_samples_path)))
            return results

        end_of_segment = int(
            orca_params.FILE_SAMPLING_SIZE_SECONDS*orca_params.LIVE_FEED_SAMPLING_RATE)

        # The features extraction should not take more than 3 seconds for 3*10 1 second segments
        features = [[segment, extract_segment_features('{}:0:{}'.format(
            (segment), (end_of_segment)))] for segment in audio_segments]

        print('Performing inference for {} audio segments'.format(
            (len(audio_segments))))

        if len(features) == 0:
            return results

        # I'm sure there is a better way to do this
        features = np.array(features)
        x = np.array([i[0] for i in features[:, 1]])

        results = model.predict(x=x,
                                batch_size=orca_params.BATCH_SIZE,
                                verbose=1)
        results = np.array(
            [[encoder.classes_[np.argmax(i)], np.max(i)] for i in results])
        # Add the filenames
        file_names = features[:, 0]
        file_names = np.array([os.path.basename(file_name) for file_name in file_names])
        results = np.hstack((file_names.reshape(len(file_names), 1), results))

        results = results[(results[:, [2]].astype(float) > probability_threshold).ravel()
                                   & (results[:, [1]] != orca_params.NOISE_CLASS).ravel()]

        if len(results) > 0:
            destination_folder = os.path.join(
                orca_params.DETECTIONS_PATH, POSITIVE_INFERENCE_TIMESTAMP)
            shutil.copytree(inference_samples_path, destination_folder)
            print(f'Copied positive inference results at:{destination_folder}')
            pd.DataFrame(results).to_csv(os.path.join(destination_folder, "results.csv"))

        shutil.rmtree(inference_samples_path)
    except KeyboardInterrupt:
        print('Received CTRL-C request to abort. BYE!')
        sys.exit(1)

# TODO: catch a specific exception here.  Otherwise we can't exit with sys.exit().
#     except:
#         print('Unable to perform inference for {}'.format(
#             (inference_samples_path)))

    return results


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
              default=orca_params.ORCASOUND_DEFAULT_STREAM_NAME,
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
@click.option('--label-encoder-path',
              help='Specify the label encoder path to use.',
              default=os.path.join(orca_params.OUTPUT_PATH,
                                   'label_encoder_latest.p'),
              show_default=True)
@click.option('--weights-path',
              help='Specify the weights path to use.',
              default=os.path.join(orca_params.OUTPUT_PATH, orca_params.DEFAULT_MODEL_NAME,
                                   'weights.best.hdf5'),
              show_default=True)
@click.option('--probability-threshold',
              type=float,
              help='Specify the minimum inference probability for the positive results.',
              default=orca_params.LIVE_FEED_MINIMUM_INFERENCE_PROBABILITY,
              show_default=True)
@click.option('--verbose',
              help='Sets the ffmpeg logs verbosity.',
              show_default=True,
              is_flag=True,
              default=False)
def live_feed_inference(model_name,
                        stream_name,
                        segment_seconds,
                        sleep_seconds,
                        iteration_seconds,
                        label_encoder_path,
                        weights_path,
                        probability_threshold,
                        verbose,
                        live_feed_path=orca_params.LIVE_FEED_PATH,
                        positive_input_samples_path=orca_params.POSITIVE_INPUT_PATH):
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

    model, encoder = create_network(
        model_name, label_encoder_path, weights_path)

    counter = 0

    while True:
        mix_with = ''
        positive_samples = glob.glob(
            os.path.join(positive_input_samples_path, '*.wav'))

        if len(positive_samples) > 0:
            mix_with = positive_samples[0]

        counter = counter + 1

        threads = []
        for _stream_name, _stream_base in orca_params.ORCASOUND_STREAMS.items():
            if stream_name != 'All' and stream_name != _stream_name:
                continue

            try:
                # get the ID of the latest stream and build URL to load
                latest = f'{_stream_base}/latest.txt'
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
                    os.makedirs(recording_samples_path)
                if not os.path.exists(inference_samples_path):
                    os.makedirs(inference_samples_path)

                thread = Thread(target=_save_audio_segments, args=(stream_url,
                                                                   _stream_name,
                                                                   segment_seconds,
                                                                   iteration_seconds,
                                                                   mix_with,
                                                                   recording_samples_path,
                                                                   verbose, ))
                threads.append(thread)
                thread.start()
            except:
                print(f'Unable to load stream from {stream_url}')

        results = _perform_inference(model, encoder, inference_samples_path, probability_threshold)
        if len(results) > 0:
            print(results)

        _ = [t.join(orca_params.LIVE_FEED_ITERATION_SECONDS) for t in threads]

        if os.path.exists(mix_with):
            os.remove(mix_with)
            print(f'{mix_with} deleted.')

        if sleep_seconds > 0:
            print(f'Sleeping for {sleep_seconds} seconds before starting next interation.\n')
            time.sleep(sleep_seconds)
        
