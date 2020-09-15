import tensorflow as tf
import tensorflow.keras.backend as K
import logging

log = logging.getLogger(__name__)


class LrScheduler:
    """Custom LR scheduler for GANs."""

    def __init__(self, lr_decay_method, max_n_steps, decay_steps_start):
        """Initialize the scheduler.

        Args:
            lr_decay_method: decay method to use.
            max_n_steps: maximum steps to run.
            decay_steps_start: step on which to start with decay.
        """
        self.lr_decay_method = lr_decay_method
        self.max_n_steps = max_n_steps
        self.decay_steps_start = decay_steps_start

    def _calculate_lr(self, step, lr):
        """Since we cannot use fit, we have to update the learning rate manually.

        Args:
            step (int): current epoch.
            lr (float): starting lr for a given model.
        """
        if step > self.decay_steps_start:
            if self.lr_decay_method == "linear":
                current_lr = lr - (step - self.decay_steps_start) * (
                    lr / (self.max_n_steps - self.decay_steps_start)
                )
            elif self.lr_decay_method == "gradient":
                current_lr = lr * 0.95 ** (step - self.decay_steps_start)
            else:
                current_lr = lr
        else:
            current_lr = lr
        return current_lr

    def update_learning_rate(self, step, lr, optimizers):
        """Since we cannot use fit, we have to update the learning rate manually.

        Args:
            step (int): current step.
            lr: current learning rate
            optimizers: list of optimizers to change
        """
        cur_lr = self._calculate_lr(step, lr)

        for _optimizer in optimizers:
            K.set_value(_optimizer.learning_rate, cur_lr)


class PatchGanLabels:
    """Generate patchgan labels for a number of methods."""

    def __init__(self, label_type):
        """Initialize the PatchGan label maker."""
        self.label_type = label_type
        self.label_values_fixed = [0, 0.9]
        self.label_values_noisy = [(0, 0.3), (0.7, 1)]

    def create_labels(self, disc_patch, true_label):
        """Create the label patch.

        Args:
            disc_patch: patch shape
            true_label: whether the label is true or false.

        Returns:
            Tensor with labels in correct shape and format.
        """
        if self.label_type == "fixed":
            if true_label:
                high = self.label_values_fixed[1]
                low = self.label_values_fixed[1] - 0.01
            else:
                high = self.label_values_fixed[0] + 0.01
                low = self.label_values_fixed[0]
            label = tf.random.uniform(minval=low, maxval=high, shape=disc_patch)
        elif self.label_type == "noisy":
            if true_label:
                high = self.label_values_noisy[1][1]
                low = self.label_values_noisy[1][0]
            else:
                high = self.label_values_noisy[0][1]
                low = self.label_values_noisy[0][0]
            label = tf.random.uniform(minval=low, maxval=high, shape=disc_patch)
        else:
            label = None
            log.error("Unknown label type.")
            exit()
        return label
