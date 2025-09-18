# AI-Clinic
Phase 1 — 0–2 hours: skeleton + data

1.	Stand up a local FHIR server or static FHIR bundle store (HAPI server or simple file store). Load Synthea bundles. HAPI FHIR+1
2.	Wire a tiny API endpoint: POST /query { query_text, patient_id }.

Phase 2 — 2–6 hours: retrieval & indexing (core)

3. Parse FHIR resources for the patient: Observation (labs), Condition (diagnoses), MedicationRequest, Procedure, Immunization, Allergies, Encounter notes. Convert those into short text snippets to index (e.g., “A1c: 9.2% (2025-09-10) — Observation/obs123”).
4. Index those snippets into a vector DB (FAISS local or Pinecone). This gives you fast retrieval of relevant datapoints for any clinician query.
Phase 3 — 6–10 hours: LLM + RAG + evidence
5. Build a retrieval step: use query → vector DB → top N snippets (e.g., top 5). Return snippets + resource ids as the “context”.
6. Compose a controlled LLM prompt that:
  •	Only uses the provided context (the returned snippets) + optionally short PubMed abstracts.
  •	Extracts the most relevant datapoints used and lists them.
  •	Produces 3 care options: each option must have a 1-line rationale and 1–2 evidence citations (PMID/URL).
7.	For evidence fetching: after the LLM suggests keywords (e.g., “SGLT2 for CKD and diabetes”), call PubMed E-utilities to search & grab 1–2 top citations (title + PMID + URL). Show those with each option. NCBI

Phase 4 — 10–14 hours: UI, explainability, QA

8. UI: Show 3 cards: (title), short recommendation, evidence (linked), and a collapsible “Used data” section listing exact FHIR resources used (id & snippet). This is the explainability piece clinicians love.
9. Add simple confidence signals: “High / Medium / Low” based on overlap of retrieved snippets and presence of guideline-level evidence.

Phase 5 — 14+ hours: polish & prepare demo

10. Prepare 2–3 demo patient scenarios: acute, chronic, and ambiguous. For each, script what the clinician asks and how the copilot answers.

