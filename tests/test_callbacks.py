import pytest

from synfire.callbacks import PrintCallback


@pytest.mark.parametrize("log_every", [True, 1.5, "10"])
def test_print_callback_rejects_non_integer_log_interval(log_every):
    with pytest.raises(ValueError, match="log_every must be an integer"):
        PrintCallback(log_every=log_every)
