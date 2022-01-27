import subprocess
from sklearn.metrics import confusion_matrix
import tensorflow as tf


def shell_exec(command_as_string):
    """
    Executes the command in shell script. Avoid using the yes | command pattern as that seems to cause an out of memory
    issue.

    Args:
        command_as_string: just a string, as if you were typing it into a  terminal

    Returns:
        stdout, stderr
    """
    try:

        process = subprocess.Popen(
            command_as_string.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False
        )
        stdout, stderr = process.communicate()

    except subprocess.CalledProcessError as e:
        stdout = e.stdout
        stderr = e.stderr

    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    return stdout, stderr


def get_confusion_matrix(val_ds, config, model):

    y_true = []
    for example in val_ds:
        y_true.append(example[1])

    y_true = tf.concat(y_true, axis=0)
    y_true = tf.argmax(y_true, axis=1)

    y_pred = model.predict(val_ds, batch_size=config.hyperparameters.batch_size, verbose=True)

    y_pred = tf.argmax(y_pred, axis=1)

    matrix = confusion_matrix(y_true, y_pred)
    acc_by_class = matrix.diagonal() / matrix.sum(axis=1)

    print({i: acc_by_class[i] for i in range(len(acc_by_class))})

    return matrix
