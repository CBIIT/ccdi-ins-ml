import pandas as pd
import datetime
import logging
from sentence_transformers import SentenceTransformer, util

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("data_processing.log"), logging.StreamHandler()]
)

# Directory setup
dir = "2.0.0.4_test"
logging.info(f"Loading data from directory: {dir}")

# Load datasets and program data from the provided CSV files
datasets_df = pd.read_csv(f'./data/input_data/{dir}/dbgap_datasets.csv')
programs_df = pd.read_csv(f'./data/input_data/{dir}/program.csv')
project_df = pd.read_csv(f'./data/input_data/{dir}/project.csv')
grant_df = pd.read_csv(f'./data/input_data/{dir}/grant.csv')

# Prepare lists to store results
program_results, project_results, grant_results = [], [], []

# Load a pre-trained sentence transformer model
logging.info("Loading sentence transformer model")
model = SentenceTransformer(
    "dunzhang/stella_en_400M_v5",
    trust_remote_code=True,
    device="cpu",
    config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
)
model.max_seq_length = 32768
model.tokenizer.padding_side = "right"

doSemantic = False


# Process each dataset row
for _, dataset_row in datasets_df.iterrows():
    # Get and format dataset data
    dataset_title = dataset_row.get('dataset_title')
    dataset_description = dataset_row.get('description', '').lower()
    dataset_funding_source = dataset_row.get('funding_source')
    dataset_pi = dataset_row.get('PI_name', '').strip().lower()

    dataset_funding_source_list = [
        fs.strip().lower() for fs in str(dataset_funding_source).split(';') if fs.strip()
    ] if pd.notna(dataset_funding_source) else []

    # Program Matching
    for _, program_row in programs_df.iterrows():
        # Extract and format program data
        program_name = program_row.get('program_name')
        program_id = program_row.get('program_id')
        program_nofo_list = [
            nofo.strip().lower() for nofo in str(program_row.get('nofo', '')).split(';') if nofo.strip()
        ]
        program_awards_list = [
            award.strip().lower() for award in str(program_row.get('award', '')).split(';') if award.strip()
        ]
        program_acronym = program_row.get('program_acronym', '').lower()
        program_pi_list = [
            pi.strip().lower() for pi in str(program_row.get('contact_pi', '')).split(';') if pi.strip()
        ]

        
        # RULE 1: Funding Source Matching
        logging.info(f"RULE 1: Funding Source Matching")
        funding_related = (
            any(nofo in dataset_description or any(nofo in fs for fs in dataset_funding_source_list) for nofo in program_nofo_list) or
            any(award in dataset_description or any(award in fs for fs in dataset_funding_source_list) for award in program_awards_list)
        )
        if funding_related:
            logging.info(f"*******************************************************************************************************")
            logging.info(f"Funding source match found between dataset and  program")
            logging.info(f"dataset_description:            '{dataset_description}'")
            logging.info(f"dataset_funding_source_list:    '{dataset_funding_source_list}'")
            logging.info(f"dataset_funding_source_list:    '{dataset_funding_source_list}'")
            logging.info(f"nofo:                           '{program_nofo_list}'")
            logging.info(f"award:                          '{program_awards_list}'")


        # RULE 2: Name or Acronym Matching
        logging.info(f"RULE 2: Name or Acronym Matching")
        name_related = any(keyword in dataset_description or keyword in dataset_title.lower() for keyword in [program_name, program_acronym])
        if name_related:
            logging.info(f"*******************************************************************************************************")
            logging.info(f"Name/acronym match found between  dataset and  program")
            logging.info(f"dataset_description:            '{dataset_description}'")
            logging.info(f"dataset_title:                  '{dataset_title}'")
            logging.info(f"program_name:                   '{program_name}'")
            logging.info(f"program_acronym:                '{program_acronym}'")

        # RULE 3: PI Matching
        logging.info(f"RULE 3: PI Matching")
        pi_related = False
        if not (len(dataset_pi) == 0 or len(program_pi_list) == 0):
            pi_related = any(pi in dataset_pi for pi in program_pi_list)
        if pi_related:
            logging.info(f"***************************************************************************************************************")
            logging.info(f"PI match found between dataset  and  program")
            logging.info(f"dataset_pi:                      '{dataset_pi}'")
            logging.info(f"program_pi_list:                 '{program_pi_list}'")


        # Append results
        program_results.append({
            'datasets': dataset_title,
            'program': program_name,
            'program_id': program_id,
            'Funding Source Matching': 'yes' if funding_related else 'no',
            'Acronym/Name Matching': 'yes' if name_related else 'no',
            'PI Matching': 'yes' if pi_related else 'no',
        })

    # Project Matching
    for _, project_row in project_df.iterrows():
        # Extract and format project data
        project_org_name = project_row.get('project_org_name', '')
        project_title = project_row.get('project_title')
        project_abstract_text = project_row.get('project_abstract_text', '')
        project_id = project_row.get('project_id')
        program_id = project_row.get('program.program_id')

        # RULE 4: Org Matching
        logging.info(f"RULE 4: Org Matching")
        org_related = project_org_name.strip().lower() in dataset_description.strip().lower() if pd.notna(project_org_name) else False
        if org_related:
            logging.info(f"*******************************************************************************************************")
            logging.info(f"Organization match found between dataset  and  project")
            logging.info(f"dataset_title:                 '{dataset_title}'")
            logging.info(f"project_title:                 '{project_title}'")


        # RULE 5: Description Semantic Matching
        logging.info(f"Description Semantic Matching ----- skip :{ not doSemantic }")
        desc_related = False
        if pd.notna(project_abstract_text) and doSemantic:
            embedding1 = model.encode(project_abstract_text + project_title)
            embedding2 = model.encode(dataset_description)
            similarity = util.cos_sim(embedding1, embedding2).item()
            desc_related = similarity > 0.6
            if desc_related:
                logging.info(f"*******************************************************************************************************")
                logging.info(f"Description semantic match found  between dataset  and  project with similarity score {similarity:.2f}")
                logging.info(f"dataset_description:             '{dataset_description}'")
                logging.info(f"project_abstract_text:           '{project_abstract_text}'")
        # Append results
        project_results.append({
            'datasets': dataset_title,
            'program_id': program_id,
            'project_id': project_id,
            'Description Matching': 'yes' if desc_related else 'no',
            'Org Matching': 'yes' if org_related else 'no',
        })

    # Grant Matching
    for _, grant_row in grant_df.iterrows():
        # Extract and format grant data
        grant_id = grant_row.get('grant_id')
        project_id = grant_row.get('project.project_id')
        principal_investigators_list = [
            pi.strip().lower() for pi in str(grant_row.get('principal_investigators', '')).split(';') if pi.strip()
        ]
        grant_opportunity_number = grant_row.get('grant_opportunity_number')
        grant_org_name = grant_row.get('grant_org_name', '') 
        grant_org_name= grant_org_name.strip().lower() if pd.notna(grant_org_name) else ''
        
        # RULE 6: PI Matching
        logging.info(f"RULE 6: PI Matching")
        grant_pi_related = False
        if not (dataset_pi != "" or len(principal_investigators_list) == 0):
            grant_pi_related = any(pi in dataset_pi for pi in principal_investigators_list)
        if grant_pi_related:
            logging.info(f"*******************************************************************************************************")
            logging.info(f"PI match found between dataset and grant")
            logging.info(f"dataset_pi:                      '{dataset_pi}'")
            logging.info(f"principal_investigators_list:    '{principal_investigators_list}'")

        # RULE 7: Funding Matching
        logging.info(f"RULE 7: Funding Matching")
        grant_funding_related = False
        if grant_opportunity_number != "":
            grant_funding_related = (
                grant_opportunity_number in dataset_funding_source
            ) if pd.notna(grant_opportunity_number) and pd.notna(dataset_funding_source) else False
        if grant_funding_related:
            logging.info(f"*******************************************************************************************************")
            logging.info(f"Funding source match found between dataset and grant")
            logging.info(f"dataset_funding_source:          '{dataset_funding_source}'")
            logging.info(f"grant_opportunity_number:        '{grant_opportunity_number}'")

        # RULE 8: Org Matching
        logging.info(f"RULE 8: Org Matching")
        grant_org_related = False
        if grant_org_name != "":
            grant_org_related = grant_org_name in dataset_description
        if grant_org_related:
            logging.info(f"*******************************************************************************************************")
            logging.info(f"Organization match found between dataset and grant")
            logging.info(f"grant_org_name:                  '{grant_org_name}'")
            logging.info(f"dataset_description:             '{dataset_description}'")

        # Append results
        grant_results.append({
            'datasets': dataset_title,
            'grant_id': grant_id,
            'project_id': project_id,
            'PI Matching': 'yes' if grant_pi_related else 'no',
            'Funding Matching': 'yes' if grant_funding_related else 'no',
            'Org Matching': 'yes' if grant_org_related else 'no',
        })

# Output results
now = datetime.datetime.now()
for result_type, results in zip(
    ["dataset_project_relationship", "dataset_program_relationship", "dataset_grant_relationship"],
    [project_results, program_results, grant_results]
):
    output_df = pd.DataFrame(results)
    file_name = f"./data/output_data/{result_type}_{now.strftime('%Y%m%d_%H%M%S')}.xlsx"
    output_df.to_excel(file_name, index=False)
    logging.info(f"Output saved to '{file_name}'")
