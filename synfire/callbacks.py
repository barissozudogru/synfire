"""Training callbacks for pluggable hooks into the FF layer training loop."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TrainingCallback(Protocol):
    """Protocol for objects that receive training loop events.

    Implement this protocol to observe or react to epoch-level training
    progress without modifying the training logic itself.

    Methods are called via ``hasattr`` guards in the training loop, so a class
    only needs to implement the methods it cares about. Partial implementations
    are valid.

    Example::

        class MyCallback:
            def on_epoch_end(self, epoch: int, loss: float, lr: float) -> None:
                if epoch % 10 == 0:
                    print(f"epoch {epoch}: loss={loss:.4f}")

        trained, losses = train_layer(state, x_pos, x_neg, callbacks=[MyCallback()])
    """

    def on_train_begin(self, total_epochs: int, n_samples: int) -> None:
        """Called once before the first training epoch.

        Args:
            total_epochs: Maximum number of epochs configured (upper bound;
                early stopping may reduce the actual count).
            n_samples: Number of positive training samples.
        """
        ...

    def on_epoch_end(self, epoch: int, loss: float, lr: float) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: Zero-based epoch index.
            loss: Mean loss for this epoch (average over mini-batches).
            lr: Learning rate that was used for this epoch.
        """
        ...

    def on_train_end(self, epochs_run: int, final_loss: float) -> None:
        """Called once after training completes (or early stopping fires).

        Args:
            epochs_run: Number of epochs that were actually executed.
            final_loss: Loss value of the last epoch.
        """
        ...


class PrintCallback:
    """Callback that prints training progress to stdout.

    Prints a summary line at the start of training, every ``log_every``
    epochs during training, and a final summary when training ends.

    Args:
        log_every: Print an epoch-level update every this many epochs.
            Set to 0 to suppress per-epoch output entirely (only
            on_train_begin and on_train_end will print).
        prefix: Optional string prepended to every line of output
            (e.g. a layer identifier such as ``"[layer 1]"``).

    Example::

        cb = PrintCallback(log_every=10, prefix="[layer 0]")
        trained, losses = train_layer(state, x_pos, x_neg, callbacks=[cb])
        # [layer 0] training started: max_epochs=30, n_samples=512
        # [layer 0] epoch  10/30  loss=0.6931  lr=0.050000
        # ...
        # [layer 0] training done: 30 epochs, final_loss=0.5120
    """

    def __init__(self, log_every: int = 10, prefix: str = "") -> None:
        if log_every < 0:
            raise ValueError(f"log_every must be >= 0, got {log_every}")
        self.log_every = log_every
        self.prefix = prefix
        self._total_epochs: int = 0

    def on_train_begin(self, total_epochs: int, n_samples: int) -> None:
        self._total_epochs = total_epochs
        tag = f"{self.prefix} " if self.prefix else ""
        print(f"{tag}training started: max_epochs={total_epochs}, n_samples={n_samples}")

    def on_epoch_end(self, epoch: int, loss: float, lr: float) -> None:
        if self.log_every == 0:
            return
        # epoch is 0-based; display as 1-based
        display_epoch = epoch + 1
        if display_epoch % self.log_every == 0:
            tag = f"{self.prefix} " if self.prefix else ""
            width = len(str(self._total_epochs))
            print(
                f"{tag}epoch {display_epoch:{width}d}/{self._total_epochs}"
                f"  loss={loss:.4f}  lr={lr:.6f}"
            )

    def on_train_end(self, epochs_run: int, final_loss: float) -> None:
        tag = f"{self.prefix} " if self.prefix else ""
        print(f"{tag}training done: {epochs_run} epochs, final_loss={final_loss:.4f}")
