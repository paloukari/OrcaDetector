import numpy as np
import os
import pickle
import pprint
import random
import re
import soundfile as sf

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

# project-specific imports
import orca_params
import vggish_params as params

# data path; when Docker container is run, data path on the host
# machine is expected to be mapped to /data
DATA_PATH = '/data/'

def _quantize_sample(label, file, sample_len=orca_params.FILE_MAX_SIZE_SECONDS):
    """
        Splits up a given file into non-overlapping segments of the specified length.
        Returns a list containing (label, 'file:start:frames') of each segment.
    """
    
    with sf.SoundFile(file) as wav_file:
        # more than 1 sec
        if wav_file.frames > wav_file.samplerate:
            frames = int(sample_len * wav_file.samplerate)
            file_parts = np.arange(0, wav_file.frames, frames)
            return [[label, '{}:{}:{}'.format(file, int(start), frames)] for start in file_parts]
            #return [[label, f"{file}:{int(start)}:{frames}"] for start in file_parts]
        else:
            return []

def _quantize_samples(samples):
    """
        Quantizes a list of samples (label, file path), returns a flattened list.
    """
    quantized_samples = [_quantize_sample(label, file) for [
        label, file] in samples]
    flat_quantized_samples = [
        item for sublist in quantized_samples for item in sublist]
    return flat_quantized_samples

def _onehot(labels, data_path=DATA_PATH):
    """
        Converts labels from strings to integers.  All classes that aren't in
        orca_params.CLASSES are mapped to 'Other'.
        
        Returns np.array[observations, classes] representing the labels
    """
    
    # convert to lists (zip generates tuples)
    labels = list(labels)

    # with label encoder, map all undesired classes to "Other"
    ohe_classes = orca_params.CLASSES + ['Other']
    encoder = LabelEncoder()
    encoder.fit(ohe_classes)
    
    # create a list holding the int class labels
    for i in range(len(labels)):
        if not labels[i] in orca_params.CLASSES:
            labels[i] = 'Other'
    encoded_labels = encoder.transform(labels)

    # build into a numpy array to return
    onehot_encoded_labels = np.zeros((len(encoded_labels), len(ohe_classes)))
    onehot_encoded_labels[np.arange(len(encoded_labels)), encoded_labels] = 1
    
    # save LabelEncoder so inverse transforms can be recovered
    label_encoder_file = os.path.join(data_path, 'label_encoder.p')
    # rename old files instead of overwriting.
    if os.path.exists(label_encoder_file):
        renamed_file = '{}-old'.format(label_encoder_file)
        os.rename(label_encoder_file, renamed_file)
        print('WARNING: Renamed previous file to {}'.format(label_encoder_file))
    with open(label_encoder_file, 'wb') as fp:
        pickle.dump(encoder, fp)
        print('Saved label encoder to {}'.format(label_encoder_file))
    

    return onehot_encoded_labels


def save_index_files(data_path=DATA_PATH, train_percentage=.70, validate_percentage=0.20):
    """
        Index files and create a train/val/test split.  Note that label one-hot
        encoding is *not* done at this point, nor are undesired classes converted
        to "Other".  That is done when loading the dataset.
    """
    
    train_indices_file = os.path.join(data_path, 'train_map.p')
    validate_indices_file = os.path.join(data_path, 'validate_map.p')
    test_indices_file = os.path.join(data_path, 'test_map.p')
    
    # rename old files instead of overwriting.
    if os.path.exists(train_indices_file):
        renamed_file = '{}-old'.format(train_indices_file)
        os.rename(train_indices_file, renamed_file)
        print('WARNING: Renamed previous file to {}'.format(renamed_file))

    if os.path.exists(validate_indices_file):
        renamed_file = '{}-old'.format(validate_indices_file)
        os.rename(validate_indices_file, renamed_file)
        print('WARNING: Renamed previous file to {}'.format(renamed_file))
        
    if os.path.exists(test_indices_file):
        renamed_file = '{}-old'.format(test_indices_file)
        os.rename(test_indices_file, renamed_file)
        print('WARNING: Renamed previous file to {}'.format(renamed_file))
            
    # build a defaultdict of all of the samples read from disk.
    # key will be the class label (text). Value will be a list of all file paths
    total_files = 0
    all_samples = defaultdict(list)
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        # extract audio filenames
        filenames = [filename for filename in filenames if os.path.splitext(filename)[1].lower() == '.wav']
        total_files += len(filenames)
        if len(filenames) == 0:
            continue  # try next folder
                
        # extract folder names as labels from a path that looks like:
        #   /data/MarineMammalName/1975
        path, year_folder = os.path.split(dirpath)
        _, label = os.path.split(path)
        # strip non alphanumeric characters
        label = re.sub('\W+','', label )
            
        all_samples[label].extend(
            [os.path.join(dirpath, file) for file in filenames])
        # print('Loaded data from {}, mapped to label={}'.format(dirpath, label))

    print('Loaded data from {} files. {} labels observed.'.format(total_files, len(all_samples)))
        
    train = defaultdict(list)
    validate = defaultdict(list)
    test = defaultdict(list)

    # TODO: change train/val/test split to happen *after* quantization
    # do a stratified train/val/test split
    for label, files in all_samples.items():
        if len(files) < 10:
            continue  # don't both to shuffle
        random.shuffle(files)
        num_train_files = int((len(files) + 1) * train_percentage)
        num_validate_files = int((len(files) + 1) * validate_percentage)
        train[label] = files[ : num_train_files]
        validate[label] = files[num_train_files : num_train_files + num_validate_files]
        test[label] = files[num_train_files + num_validate_files : ]

    # Create lists with each element looking like:
    #   ['SpermWhale', '/data/SpermWhale/1985/8500901B.wav']
    train_flattened = [[label, file] 
                        for label in train.keys() for file in train[label]]
    validate_flattened = [[label, file] 
                           for label in train.keys() for file in validate[label]]
    test_flattened = [[label, file] 
                       for label in train.keys() for file in test[label]]
        
    print('{} sample files in train'.format(len(train_flattened)))
    print('{} samples files in val'.format(len(validate_flattened)))
    print('{} samples files in test'.format(len(test_flattened)))
        
    train_flattened = _quantize_samples(train_flattened)
    validate_flattened = _quantize_samples(validate_flattened)
    test_flattened = _quantize_samples(test_flattened)

    print('\nAfter quantization:')
    print('{} sample segments in train'.format(len(train_flattened)))
    print('{} sample segments in val'.format(len(validate_flattened)))
    print('{} sample segments in test'.format(len(test_flattened)))
        
    with open(train_indices_file, 'wb') as fp:
        pickle.dump(train_flattened, fp)
        print('Saved training set to {}'.format(train_indices_file))

    with open(validate_indices_file, 'wb') as fp:
        pickle.dump(validate_flattened, fp)
        print('Saved validation set to {}'.format(validate_indices_file))

    with open(test_indices_file, 'wb') as fp:
        pickle.dump(test_flattened, fp)
        print('Saved test set to {}'.format(test_indices_file))


def load_dataset(data_path=DATA_PATH, dataset_type=None):
    """
        Load the lists of files and labels for train, val, or test set.  One-hot
        encoding of the labels is performed at this time based on the desired
        classes specified in orca_params.CLASSES (all others are mapped to 'Other').
        The pickled LabelEncoder() is also saved so that an inverse transform can
        later be performed with it if desired.
    """
    
    if dataset_type not in ('train', 'validate', 'test'):
        raise ValueError('ERROR: only train, validate, and test types are supported.')
        
    if dataset_type == 'train':
        indices_file = os.path.join(data_path, 'train_map.p')
    elif dataset_type == 'validate':
        indices_file = os.path.join(data_path, 'validate_map.p')
    elif dataset_type == 'test':
        indices_file = os.path.join(data_path, 'test_map.p')
    
    if os.path.exists(indices_file):
        print('\nTrying to load {} dataset from {}'.format(dataset_type, indices_file))
        with open(indices_file, 'rb') as f:
            flattened = pickle.load(f)
        print('Successfully loaded.')
    else:
        raise Exception('ERROR: run database_parser.py to generate datafiles first.')
        
    labels, files = zip(*flattened)

    return files, _onehot(labels)

    
if __name__ == '__main__':

    # generate and save index files
    save_index_files()
    
    # load them to validate
    
    train_files, train_labels = load_dataset(dataset_type='train')
    print('\nSample training files:')
    pprint.pprint(train_files[:5])
    print('Training set, counts by label:')
    print(np.sum(train_labels, axis=0))
    
    val_files, val_labels = load_dataset(dataset_type='validate')
    print('\nSample validation files:')
    pprint.pprint(val_files[:5])
    print('Validation set, counts by label:')
    print(np.sum(val_labels, axis=0))
    
    test_files, test_labels = load_dataset(dataset_type='test')
    print('\nSample test files:')
    pprint.pprint(test_files[:5])
    print('Test set, counts by label:')
    print(np.sum(test_labels, axis=0))

    