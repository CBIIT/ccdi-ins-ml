# Functions for formatting strings in this project

import re

def clean_award_number(award_number):
    """Cleans the award number

    Args:
        award_number (str): Award number
    """

    cleaned_award_number = re.sub(r'[\s]+', '', award_number) # Remove spaces
    cleaned_award_number = re.sub(r'^\d', '', cleaned_award_number) # Remove leading digit
    cleaned_award_number = re.sub(r'-([^-]{2}|[^-]{4})$', '', cleaned_award_number) # Remove everything after the last hyphen
    cleaned_award_number = re.sub(r'[-]+', '', cleaned_award_number) # Remove hyphens
    cleaned_award_number = re.sub(r'^RO1', 'R01', cleaned_award_number)  # Correct the "O" in "RO1" to a zero at the start

    return cleaned_award_number
