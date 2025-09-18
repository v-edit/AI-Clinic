from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fhir.resources.patient import Patient

# Load ClinicalBERT model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to parse relevant patient info from FHIR JSON
def parse_fhir_record(fhir_json):
    patient = Patient(**fhir_json.get("patient", {}))
    patient_name = patient.name[0].given[0] if patient.name else "Unknown"
    # Extract other relevant clinical info here if needed
    return {"patient_name": patient_name}

# FastAPI app and request body schema
class QueryRequest(BaseModel):
    fhir_record: dict
    question: str

app = FastAPI()

@app.post("/clinical-query")
async def clinical_query(req: QueryRequest):
    # Parse patient info from FHIR record
    patient_info = parse_fhir_record(req.fhir_record)
    
    # Create input text combining patient data and clinical question
    text_input = f"Patient name: {patient_info['patient_name']}. Question: {req.question}"
    
    # Tokenize input and run ClinicalBERT model
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    
    # Extract model outputs (logits here, adjust based on use case)
    logits = outputs.logits.detach().cpu().numpy().tolist()
    
    # Return raw logits (replace with further processing as needed)
    return {"response_logits": logits}
