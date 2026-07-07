# Functions for comparing strings in this project

import re

def does_match_str(target, str):
    """ Searches for target in string

    Args:
        target (str): The substring to find
        str (str): String in which to look for the substring
    """

    # Ensure that the pattern has word boundaries
    pattern = fr'\b{target}\b'

    return bool(re.findall(pattern, str))
