import logging
from difflib import SequenceMatcher
from nameparser import HumanName
from util.formatting import (
    clean_award_number
)
from util.comparing import (
    does_match_str
)

logger = logging.getLogger(__name__)

def get_dataset_program_funding_matches(dataset_description,
        dataset_funding_source_list, dataset_funding_source_list_cleaned,
        program_awards_list, program_awards_list_cleaned,
        program_nofo_list, program_nofo_list_cleaned):
    """Checks whether the dataset and program have matching funding

    Args:
        dataset_description (str): The dataset's description
        dataset_funding_source_list (list): The dataset's funding sources
        program_awards_list (list): The program's awards
        program_nofo_list (list): The program's notices of funding opportunities
    """

    matches = {}
    difflib_matches = {
        'award': {
            'funding_source': {
                'highest_ratio': 0.0,
                'dataset_str': None,
                'program_str': None
            }
        }
    }

    for award in (program_awards_list + program_awards_list_cleaned):
        award_strs = [award]

        if len(award) > 3:
            award_strs.append(award[3:]) # Alternate award string with first three characters removed

        for award_str in award_strs:
            if not award_str:
                continue

            # Skip purely numerical award numbers
            if award_str.isdigit():
                continue

            if award_str not in dataset_description:
                continue

            if 'awards_to_desc' not in matches:
                matches['awards_to_desc'] = {
                    'program_awards': [],
                    'dataset_description': dataset_description
                }

            matches['awards_to_desc']['program_awards'].append(award_str)

        for funding_source in (dataset_funding_source_list + dataset_funding_source_list_cleaned):
            funding_source_strs = [funding_source, clean_award_number(funding_source)]

            for funding_source_str in funding_source_strs:
                for award_str in award_strs:
                    # Skip purely numerical award numbers
                    if award_str.isdigit():
                        continue

                    difflib_fs_similarity = SequenceMatcher(None, award_str, funding_source_str).ratio()

                    if difflib_fs_similarity > difflib_matches['award']['funding_source']['highest_ratio']:
                        difflib_matches['award']['funding_source']['highest_ratio'] = difflib_fs_similarity
                        difflib_matches['award']['funding_source']['dataset_str'] = funding_source_str
                        difflib_matches['award']['funding_source']['program_str'] = award_str

                    if award_str not in funding_source_str:
                        continue

                    if 'awards_to_fs' not in matches:
                        matches['awards_to_fs'] = []

                    matches['awards_to_fs'].append({
                        'program_award': award_str,
                        'dataset_funding_source': funding_source_str
                    })

    for nofo in (program_nofo_list + program_nofo_list_cleaned):
        if nofo in dataset_description:
            if "nofos_to_desc" not in matches:
                matches["nofos_to_desc"] = {
                    "nofos": [],
                    "dataset_description": dataset_description
                }

            matches["nofos_to_desc"]["nofos"].append(nofo)

        for funding_source in (dataset_funding_source_list + dataset_funding_source_list_cleaned):
            if nofo not in funding_source:
                continue

            matches["nofos_to_fs"] = []

            matches["nofos_to_fs"].append({
                "program_nofo": nofo,
                "dataset_funding_source": funding_source
            })

    if matches:
        logger.info(f"*******************************************************************************************************")
        logger.info(f"Funding source match found between dataset and  program")
        logger.info(f"dataset_description:            '{dataset_description}'")
        logger.info(f"dataset_funding_source_list:    '{dataset_funding_source_list}'")
        logger.info(f"nofo:                           '{program_nofo_list}'")
        logger.info(f"award:                          '{program_awards_list}'")

    return (matches, difflib_matches)

def get_dataset_program_name_matches(dataset_description, dataset_title, program_acronym, program_name):
    """Checks whether the dataset mentions the program's acronym or name

    Args:
        dataset_description (str): The dataset's description
        dataset_title (str): The dataset's title
        program_acronym (str): The program's acronym
        program_name (str): The program's name
    """

    matches = {}
    difflib_matches = {
        'acronym': {
            'description': {
                'highest_ratio': 0.0,
                'acronym_str': None,
                'description_str': None
            },
            'title': {
                'highest_ratio': 0.0,
                'acronym_str': None,
                'title_str': None
            }
        },
        'name': {
            'description': {
                'highest_ratio': 0.0,
                'name_str': None,
                'description_str': None
            },
            'title': {
                'highest_ratio': 0.0,
                'name_str': None,
                'title_str': None
            }
        }
    }

    if does_match_str(program_acronym, dataset_description):
        matches["acr_to_desc"] = {
            "program_acronym": program_acronym,
            "dataset_description": dataset_description
        }

    if does_match_str(program_acronym, dataset_title):
        matches["acr_to_title"] = {
            "program_acronym": program_acronym,
            "dataset_title": dataset_title
        }

    if program_name.lower() in dataset_description.lower():
        matches["name_to_desc"] = {
            "program_name": program_name,
            "dataset_description": dataset_description
        }

    if program_name.lower() in dataset_title.lower():
        matches["name_to_title"] = {
            "program_name": program_name,
            "dataset_title": dataset_title
        }

    # Difflib comparisons
    difflib_acr_desc_score = SequenceMatcher(None, program_acronym, dataset_description).ratio()
    difflib_acr_title_score = SequenceMatcher(None, program_acronym, dataset_title).ratio()
    difflib_name_desc_score = SequenceMatcher(None, program_name, dataset_description).ratio()
    difflib_name_title_score = SequenceMatcher(None, program_name, dataset_title).ratio()

    if difflib_acr_desc_score > difflib_matches['acronym']['description']['highest_ratio']:
        difflib_matches['acronym']['description']['highest_ratio'] = difflib_acr_desc_score
        difflib_matches['acronym']['description']['acronym_str'] = program_acronym
        difflib_matches['acronym']['description']['description_str'] = dataset_description

    if difflib_acr_title_score > difflib_matches['acronym']['title']['highest_ratio']:
        difflib_matches['acronym']['title']['highest_ratio'] = difflib_acr_title_score
        difflib_matches['acronym']['title']['acronym_str'] = program_acronym
        difflib_matches['acronym']['title']['title_str'] = dataset_title

    if difflib_name_desc_score > difflib_matches['name']['description']['highest_ratio']:
        difflib_matches['name']['description']['highest_ratio'] = difflib_name_desc_score
        difflib_matches['name']['description']['name_str'] = program_name
        difflib_matches['name']['description']['description_str'] = dataset_description

    if difflib_name_title_score > difflib_matches['name']['title']['highest_ratio']:
        difflib_matches['name']['title']['highest_ratio'] = difflib_acr_desc_score
        difflib_matches['name']['title']['name_str'] = program_name
        difflib_matches['name']['title']['title_str'] = dataset_title

    if matches:
        logger.info(f"*******************************************************************************************************")
        logger.info(f"Name/acronym match found between  dataset and  program")
        logger.info(f"dataset_description:            '{dataset_description}'")
        logger.info(f"dataset_title:                  '{dataset_title}'")
        logger.info(f"program_name:                   '{program_name}'")
        logger.info(f"program_acronym:                '{program_acronym}'")

    return (matches, difflib_matches)

def get_dataset_program_pi_matches(dataset_pi, program_pi_list):
    """Checks whether the dataset and program have matching PIs

    Args:
        dataset_pi (str): The dataset's list of PIs
        program_pi_list (list): The program's list of PIs
    """

    dataset_name_tuples = [(name.first, name.last) for name in map(HumanName, dataset_pi)]
    program_name_tuples = [(name.first, name.last) for name in map(HumanName, program_pi_list)]
    common = set(dataset_name_tuples) & set(program_name_tuples)

    if common:
        logger.info(f"***************************************************************************************************************")
        logger.info(f"PI match found between dataset  and  program")
        logger.info(f"dataset_pi:                      '{dataset_pi}'")
        logger.info(f"program_pi_list:                 '{program_pi_list}'")

    return ', '.join(f'{first} {last}' for first, last in common)
