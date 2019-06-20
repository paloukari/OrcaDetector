"""
Utility functions for the Orca project.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton

Adapted from: https://github.com/mwinton/w266-final-project
"""

# import matplotlib this way to run without a display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# project-specific imports
import orca_params

def plot_train_metrics(model_history, run_timestamp='unspecified'):
    """
        Generate and save figure with plot of train and validation losses.
        Currently, plot_type='epochs' is the only option supported.
        Args:
            model_history: Keras History instance; History.history dict contains a list of metrics per epoch
            run_timestamp: time of the run, to be used in naming saved artifacts from the same run
        Returns:
            loss_fig_path: string representing path to the saved loss plot for the run
            acc_fig_path: string represeenting path to the saved accuracy plot for the run
    """

    # extract data from history dict
    train_losses = model_history.history['loss']
    val_losses   = model_history.history['val_loss']
    train_acc    = model_history.history['acc']
    val_acc      = model_history.history['val_acc']

    # define filenames
    loss_fig_path = os.path.join(orca_params.OUTPUT_PATH, 'orca_train_loss_{}.png'.format(run_timestamp))
    acc_fig_path  = os.path.join(orca_params.OUTPUT_PATH, 'orca_train_acc_{}.png'.format(run_timestamp))

    # generate and save loss plot
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xticks(np.arange(0, len(train_losses), step=1))
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title('Orca Model\nRun time: {}'.format(run_timestamp))
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
    plt.title('Orca Model\nRun time: {}'.format(run_timestamp))
    plt.legend(('Training', 'Validation'))
    plt.savefig(acc_fig_path)

    return loss_fig_path, acc_fig_path


def save_model(model, run_timestamp='unspecified'):
    """
        Generate and save figure with plot of train and validation losses.
        Currently, plot_type='epochs' is the only option supported.
        Args:
            model: trained Keras Model
            run_timestamp: time of the run, to be used in naming saved artifacts from the same run
        Returns:
            json_path: string representing path to the saved model config json file
            weights_path: string representing path to the saved model weights
    """
    
    # save model config
    output_json = os.path.join(orca_params.OUTPUT_PATH, 'orca_config_{}.json'.format(run_timestamp))
    with open(output_json, 'w') as json_file:
        json_file.write(model.to_json())

    # save trained model weights
    output_weights = os.path.join(orca_params.OUTPUT_PATH, 'orca_weights_{}.hdf5'.format(run_timestamp))
    model.save_weights(output_weights)
    
    # Create symbolic link to the most recent weights (to use for testing)
    symlink_path = os.path.join(orca_params.OUTPUT_PATH, 'orca_weights_latest.hdf5')
    try:
        os.symlink(output_weights, symlink_path)
    except FileExistsError:
        # If the symlink already exist, delete and create again
        os.remove(symlink_path)
        os.symlink(output_weights, symlink_path)
    print('Created symbolic link to final weights -> {}'.format(symlink_path))
    
    return output_json, output_weights


