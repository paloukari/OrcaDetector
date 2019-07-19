# -*- coding: future_fstrings -*-

"""
Utility functions for the Orca project.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton

Adapted from: https://github.com/mwinton/w266-final-project
"""

# import matplotlib this way to run without a display
import orca_params
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')

from sklearn.metrics import classification_report, confusion_matrix

def calculate_accuracies(results, labels=None, run_timestamp='unspecified'):
    """
        Calculate, displays, and savees various accuracy metrics after a test run.  This method is
        only for post-processing; Keras reports its built-in accuracy calculations during
        training/validation runs.
        Args:
            results: list, with each list item being a dict of name/value pairs
            labels: list of labels
        Returns:
            no return value
    """

    if labels is None:
        return

    print(classification_report(labels, results))

    # also save classification report and confusion matrix to disk
    clf_report = classification_report(labels, results, output_dict=True)
    clf_filename = f'classification_report_{run_timestamp}.json'
    clf_path = os.path.join(orca_params.OUTPUT_PATH, clf_filename)
    df = pd.DataFrame(clf_report)
    df.to_json(clf_path, orient='columns')
    print(f'Classification report saved to {clf_path}')

    conf_matrix = confusion_matrix(labels, results)
    conf_matrix_filename = f'confusion_matrix_{run_timestamp}.csv'
    conf_matrix_path = os.path.join(orca_params.OUTPUT_PATH,
                                    conf_matrix_filename)
    df = pd.DataFrame(conf_matrix)
    df.to_csv(conf_matrix_path, index=True)
    print(f'Confusion matrix saved to {conf_matrix_path}')

    
def create_or_replace_symlink(file_path, symlink_path):
    """
        Gracefully delete and recreate if it already exists.  The built-in
        os.symlink() function doesn't allow for replacing existing links.

        ARGS:
            file_path = path of file to be linked to
            symlink_path = path of the symlink to be created

        RETURNS:
            nothing
    """

    try:
        os.symlink(file_path, symlink_path)
    except FileExistsError:
        os.remove(symlink_path)
        os.symlink(file_path, symlink_path)


def plot_train_metrics(model_history, output_folder):
    """
        Generate and save figure with plot of train and validation losses.
        Currently, plot_type='epochs' is the only option supported.
        Args:
            model_history: Keras History instance; History.history dict contains a list of metrics per epoch
            output_folder: The output folder
        Returns:
            loss_fig_path: string representing path to the saved loss plot for the run
            acc_fig_path: string represeenting path to the saved accuracy plot for the run
    """

    # extract data from history dict
    train_losses = model_history.history['loss']
    val_losses = model_history.history['val_loss']
    best_val_loss = min(val_losses)

    train_acc = model_history.history['acc']
    val_acc = model_history.history['val_acc']
    best_val_acc = val_acc[np.argmin(val_losses)]

    # define filenames
    loss_filename = f'orca_loss_plot_val_loss_{best_val_loss:.4f}_val_acc_{best_val_acc:.4f}.png'
    loss_fig_path = os.path.join(output_folder, loss_filename)
    acc_filename = f'orca_accuracy_plot_val_loss_{best_val_loss:.4f}_val_acc_{best_val_acc:.4f}.png'
    acc_fig_path = os.path.join(output_folder, acc_filename)

    # generate and save loss plot
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xticks(np.arange(0, len(train_losses), step=1))
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title(f'Orca Model\nRun time: {os.path.basename(output_folder)}')
    plt.legend(('Training', 'Validation'))
    plt.savefig(loss_fig_path)

    # clear axes and figure to reset for next plot
    plt.cla()
    plt.clf()

    # generate and save accuracy plot
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.xticks(np.arange(0, len(train_acc), step=1))
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.title(f'Orca Model\nRun time: {os.path.basename(output_folder)}')
    plt.legend(('Training', 'Validation'))
    plt.savefig(acc_fig_path)

    return loss_fig_path, acc_fig_path


def save_model_config(model, model_history, output_folder):
    """
        Generate and save figure with plot of train and validation losses.
        Currently, plot_type='epochs' is the only option supported.
        Args:
            model: trained Keras Model
            model_history: Keras History object, used to extract loss for file names.
            output_folder: The output folder
        Returns:
            json_path: string representing path to the saved model config json file
    """

    # extract data from history dict
    val_losses = model_history.history['val_loss']
    val_acc = model_history.history['val_acc']
    best_val_loss = min(val_losses)
    best_val_acc = val_acc[np.argmin(val_losses)]
    
    # save model config
    json_filename = f'config_val_loss_{best_val_loss}_val_acc_{best_val_acc}.json'
    output_json = os.path.join(
        output_folder, json_filename)
    with open(output_json, 'w') as json_file:
        json_file.write(model.to_json())
    print(f'Keras model config written to {output_json}')

    return output_json