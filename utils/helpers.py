"""
helpers.py
----------
Generic, reusable helper functions with no business logic attached.

Only a minimal, framework-agnostic helper is included at this stage
(file extension checking against config), since actual file upload
handling is not yet wired into the routes.
"""

import os


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """
    Check whether a given filename has an extension present in the
    allowed_extensions set.

    Args:
        filename (str): Name of the uploaded file (e.g., "video.mp4").
        allowed_extensions (set): Set of permitted extensions, e.g.
                                   {"mp4", "wav", "jpg"} (from config.py).

    Returns:
        bool: True if the file's extension is allowed, False otherwise.
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )
