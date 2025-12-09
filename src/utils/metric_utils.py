import numpy as np


def check_unique(items: np.ndarray) -> bool:
    """Check if all items in the array are unique.

    Args:
        items (np.ndarray): Array of items to check (shape: n_items x top_n).
    Returns:
        bool: True if all items are unique, False otherwise.
    """
    sorted_rows = np.sort(items, axis=1)

    return ~np.any(sorted_rows[:, 1:] == sorted_rows[:, :-1], axis=1).any()
