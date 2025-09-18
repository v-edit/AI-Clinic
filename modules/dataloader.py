# modules/data_loader.py

import os
import json

# This path navigates from 'modules' up to the root and then into 'data'
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'fhir_records'))

def get_patient_list():
    """Returns a list of available patient IDs from the filenames."""
    try:
        files = os.listdir(DATA_DIR)
        patient_ids = [f.replace('.json', '') for f in files if f.endswith('.json')]
        return sorted(patient_ids)
    except FileNotFoundError:
        return [] # Return empty list if directory doesn't exist

def load_patient_data(patient_id: str):
    """Loads a single patient's FHIR data from a JSON file."""
    file_path = os.path.join(DATA_DIR, f"{patient_id}.json")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return None

def format_patient_summary(patient_data):
    """Creates a simple, readable summary from FHIR data."""
    if not patient_data:
        return "Patient data not found."
    try:
        # Navigate through the FHIR structure to find patient details
        patient_resource = next(entry['resource'] for entry in patient_data.get('entry', []) if entry.get('resource', {}).get('resourceType') == 'Patient')
        name = patient_resource['name'][0]
        given_name = " ".join(name.get('given', []))
        family_name = name.get('family', '')
        gender = patient_resource.get('gender', 'N/A')
        birth_date = patient_resource.get('birthDate', 'N/A')
        
        summary = f"**Name:** {given_name} {family_name}\n\n"
        summary += f"**Gender:** {gender}\n\n"
        summary += f"**Birth Date:** {birth_date}"
        return summary
    except (KeyError, IndexError, StopIteration):
        return "Could not parse patient summary from FHIR data."