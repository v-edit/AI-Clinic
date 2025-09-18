
import os
import json

DATA_DIR = "data"  # folder where all FHIR JSON files are stored

def load_all_fhir():
    """Dynamically load all FHIR resources from all JSON files into a single virtual FHIR Bundle.
    Handles both FHIR Bundles and single resources."""
    bundle = {"resourceType": "Bundle", "type": "collection", "entry": []}
    for file in os.listdir(DATA_DIR):
        if file.endswith(".json"):
            file_path = os.path.join(DATA_DIR, file)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    # If this is a FHIR Bundle, add all its entries
                    if data.get("resourceType") == "Bundle" and "entry" in data:
                        for entry in data["entry"]:
                            if "resource" in entry:
                                bundle["entry"].append(entry)
                    else:
                        bundle["entry"].append({"resource": data})
            except Exception as e:
                print(f"❌ Error reading {file}: {e}")
    print(f"✅ Loaded {len(bundle['entry'])} resources into bundle")
    return bundle

def summarize_bundle(bundle):
    """
    Turn a FHIR bundle into a compact doctor-friendly summary.
    """
    summary = []
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        rtype = res.get("resourceType")
        if rtype == "Patient":
            name = res.get("name", [{}])[0].get("text", "Unknown")
            gender = res.get("gender", "Unknown")
            dob = res.get("birthDate", "Unknown")
            summary.append(f"Patient: {name}, Gender: {gender}, DOB: {dob}")
        elif rtype == "Condition":
            cond = res.get("code", {}).get("text", "Unknown condition")
            summary.append(f"Condition: {cond}")
        elif rtype == "MedicationRequest":
            med = res.get("medicationCodeableConcept", {}).get("text", "Unknown medication")
            summary.append(f"Medication: {med}")
        elif rtype == "Observation":
            code = res.get("code", {}).get("text")
            value = res.get("valueQuantity", {}).get("value")
            unit = res.get("valueQuantity", {}).get("unit")
            if code and value:
                summary.append(f"Lab/Observation: {code} = {value} {unit or ''}")
    return "\n".join(summary)

