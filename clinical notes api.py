# Import necessary libraries.
# Flask is used for creating the web API.
# The transformers library from Hugging Face is used for the summarization model.
# Torch is the deep learning framework that powers the model.
import torch
from flask import Flask, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer

# --- Task 1: Install and load the summarization model ---
# Note: The first time you run this, it will download the pre-trained model
# and tokenizer from the internet. This may take a few minutes.
# For a production application, you would load these objects once at startup
# outside of any function to avoid re-loading on every API call.

# Load the BART tokenizer and model.
# 'facebook/bart-large-cnn' is a pre-trained model specifically fine-tuned for
# summarization tasks on the CNN/Daily Mail dataset, making it an excellent choice.
print("Loading BART tokenizer and model...")
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
print("Model loaded successfully.")

# --- Flask Application Setup ---
app = Flask(__name__)

# --- Task 2: Implement query understanding (Keyword extraction) ---
def extract_keywords(query: str) -> list:
    """
    Extracts relevant keywords from a user's query using simple matching.
    
    Args:
        query: The input string from the user.
        
    Returns:
        A list of extracted keywords.
    """
    # Convert query to lowercase for case-insensitive matching.
    query = query.lower()
    keywords = []
    
    # Simple keyword-based logic. This can be expanded with more sophisticated
    # NLP techniques for real-world applications.
    if "chest pain" in query:
        keywords.append("chest pain")
    if "treatment" in query:
        keywords.append("treatment")
    
    return keywords

# --- Task 3: Pull relevant notes from mock FHIR JSON ---
def get_fhir_notes(keywords: list) -> str:
    """
    Mocks pulling relevant clinical notes from a FHIR-like JSON structure.
    
    In a real application, this would involve a database query or an API call
    to a service that holds electronic health records.
    
    Args:
        keywords: A list of keywords to filter the notes by.
        
    Returns:
        A single string containing all the relevant note text, joined together.
    """
    # Mock FHIR data representing a patient's clinical notes.
    fhir_data = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"text": "Patient presented with a history of recurrent headaches and migraines for 3 months."}},
            {"resource": {"text": "Patient reports severe chest pain radiating to the left arm. Suspected myocardial infarction."}},
            {"resource": {"text": "Treatment plan involves a regimen of beta-blockers and aspirin for the chest pain."}},
            {"resource": {"text": "Follow-up scheduled in 2 weeks to review lab results and vitals."}},
            {"resource": {"text": "Patient has a history of hypertension and Type 2 Diabetes."}}
        ]
    }
    
    relevant_notes = []
    for entry in fhir_data['entry']:
        note_text = entry['resource']['text'].lower()
        # Check if any of the keywords are present in the note.
        if any(keyword in note_text for keyword in keywords):
            relevant_notes.append(entry['resource']['text'])
            
    # Join the relevant notes into a single string for summarization.
    return " ".join(relevant_notes)

# --- Task 4: Pipe extracted text to the summarizer model ---
def summarize_notes(text: str) -> str:
    """
    Uses the BART model to generate a summary of the input text.
    
    Args:
        text: A string containing the combined clinical notes.
        
    Returns:
        A concise summary string.
    """
    # Tokenize the input text. The max_length is capped to fit the model's
    # input size.
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    
    # Generate the summary.
    # num_beams=4: Uses beam search for better summary quality.
    # max_length=150: Limits the output summary to a reasonable length.
    # early_stopping=True: Stops generation when a complete sentence is formed.
    summary_ids = model.generate(
        inputs['input_ids'], 
        num_beams=4, 
        max_length=150, 
        early_stopping=True
    )
    
    # Decode the generated token IDs back into a human-readable string.
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# --- Task 5: Add mock evidence suggestions ---
def get_treatment_options() -> list:
    """
    Hardcodes and returns a list of evidence-based care options with citations.
    
    In a real system, this could be a query to a knowledge base or a a
    guideline-specific database.
    
    Returns:
        A list of dictionaries, where each dictionary represents a care option.
    """
    # Hardcoded treatment options with mock citations.
    return [
        {
            "title": "Aspirin and Nitroglycerin Therapy",
            "description": "Initial treatment for suspected myocardial infarction involves administering aspirin to prevent further clot formation and nitroglycerin to dilate coronary arteries and improve blood flow.",
            "citation": "World Health Organization, Guideline on Acute Coronary Syndromes, 2021."
        },
        {
            "title": "Cardiac Catheterization",
            "description": "For confirmed cases of myocardial infarction, cardiac catheterization is a procedure to visualize the coronary arteries and clear blockages, often with the placement of a stent.",
            "citation": "PubMed, 'Role of Cardiac Catheterization in Acute Myocardial Infarction', 2022."
        },
        {
            "title": "Lifestyle Modifications and Long-Term Medications",
            "description": "Long-term management includes a combination of diet and exercise recommendations, along with prescribed medications like statins and beta-blockers to manage cholesterol and blood pressure.",
            "citation": "American Heart Association, 'Guidelines for the Management of Patients with Myocardial Infarction', 2020."
        }
    ]

# --- Main API Endpoint ---
@app.route('/query', methods=['POST'])
def handle_query():
    """
    This is the main API endpoint that processes a user's query.
    It orchestrates the entire workflow from query to final response.
    """
    try:
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "No 'query' key found in request body"}), 400
        
        # Step 1: Extract keywords from the query.
        keywords = extract_keywords(query)
        
        # Step 2: Get relevant notes based on the keywords.
        extracted_text = get_fhir_notes(keywords)
        
        if not extracted_text:
            return jsonify({"error": "No relevant notes found for the given query."}), 404
        
        # Step 3: Summarize the extracted text.
        summary = summarize_notes(extracted_text)
        
        # Step 4: Get mock treatment options with citations.
        treatment_options = get_treatment_options()
        
        # Return a structured JSON response.
        return jsonify({
            "query": query,
            "summary": summary,
            "suggested_care_options": treatment_options
        })
        
    except Exception as e:
        # Generic error handling for unexpected issues.
        return jsonify({"error": str(e)}), 500

# --- Start the Flask server ---
if __name__ == '__main__':
    # Running in debug mode allows for automatic code reloading on changes.
    # It should be set to False in a production environment.
    app.run(debug=True)