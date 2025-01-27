import logging
logger = logging.getLogger(__name__)

def get_dataset_program_name_matches(dataset_description, dataset_title, program_acronym, program_name):
    """Check whether the dataset mentions the program's acronym or name

    Args:
        dataset_description (str): The dataset's description
        dataset_title (str): The dataset's title
        program_acronym (str): The program's acronym
        program_name (str): The program's name
    """

    matches = {}

    if program_acronym in dataset_description:
        matches["acr_to_desc"] = {
            "program_acronym": program_acronym,
            "dataset_description": dataset_description
        }

    if program_acronym in dataset_title:
        matches["acr_to_title"] = {
            "program_acronym": program_acronym,
            "dataset_title": dataset_title
        }

    if program_name in dataset_description:
        matches["name_to_desc"] = {
            "program_name": program_name,
            "dataset_description": dataset_description
        }

    if program_name in dataset_title:
        matches["name_to_title"] = {
            "program_name": program_name,
            "dataset_title": dataset_title
        }

    if matches:
        logger.info(f"*******************************************************************************************************")
        logger.info(f"Name/acronym match found between  dataset and  program")
        logger.info(f"dataset_description:            '{dataset_description}'")
        logger.info(f"dataset_title:                  '{dataset_title}'")
        logger.info(f"program_name:                   '{program_name}'")
        logger.info(f"program_acronym:                '{program_acronym}'")

    return matches
