
# MediPlate AI
**AI-Powered Personalized Medication, Nutrition & Food Safety Assistant**

A Streamlit college-competition prototype combining patient-context inputs, a local retrieval-augmented generation (RAG) pipeline, Gemini text/image analysis, a medication reminder organizer, and personalized meal ideas.

## Features
- Food image upload and cautious AI-assisted identification
- Patient context: conditions, allergies, medicine names and diet preferences
- Genuine local RAG: TF-IDF retrieval over bundled source excerpts before Gemini generation
- Visible retrieved evidence with source title, publisher, relevance score and URL
- Meal ideas with source context and uncertainty
- Session-only medication reminder schedule
- Safety-focused explanations and clear limitations

## Requirements
- Python 3.10 or newer recommended
- Gemini API key for generative answers and image analysis
- Internet access for Gemini API calls
- Local knowledge retrieval works without an API key

## Run locally (Windows / VS Code)
1. Extract the ZIP.
2. Open the extracted `TeamName_MediPlateAI` folder in VS Code.
3. Open Terminal in VS Code.
4. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run:
   ```bat
   .venv\Scripts\activate.bat
   ```

5. Install packages:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Copy `.env.example` to `.env`, then replace the placeholder with your own Gemini API key:
   ```text
   GEMINI_API_KEY=your_real_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```
7. Start the app:
   ```bash
   streamlit run app.py
   ```
8. The terminal will display a local URL, normally `http://localhost:8501`.

## Demo flow (3–4 minutes)
1. Open the app and explain the demo profile is fictional.
2. Set an allergy and a condition in the sidebar.
3. Open **Ask with evidence**; ask a question that matches one of the bundled records.
4. Point to the retrieved passages and source URLs. Explain that this is the RAG retrieval stage.
5. Open **Food scanner**, upload a clear meal photo and add confirmed ingredients if known.
6. Explain the result, uncertainties, retrieved evidence and why a photo cannot prove hidden ingredients or safety.
7. Show the medication reminder form. Enter instructions copied from a fictional example prescription.
8. Finish with the **How it works** tab and limitations.

## RAG architecture
1. The app loads curated, source-attributed excerpts from `knowledge_base/trusted_medical_sources.json`.
2. `retrieve()` builds a TF-IDF representation of document title, topic and content.
3. The user's query and current profile context are compared with the local document vectors.
4. Top matching excerpts are supplied to Gemini as context.
5. The model is instructed to cite retrieved source IDs and disclose when evidence is insufficient.
6. The app visibly displays the retrieved passages and source links.

This is a lightweight demonstrator. For a production system, add stronger hybrid/semantic retrieval, expert-reviewed and versioned sources, automated evaluation, access control, encryption, audit logs, robust medication terminology matching and clinical validation.

## Safety / privacy
- This is an educational prototype, not a medical device, diagnosis tool or substitute for a clinician.
- Use fictional demo profiles only. Do not upload identifiable medical records or real patient information.
- Food photos cannot establish hidden ingredients, allergen cross-contact, spoilage, exact serving size or precise nutrition.
- Missing retrieval results do not prove an interaction or risk is absent.
- Never change or stop prescribed medicines based on this app. Verify important concerns with a doctor or pharmacist.
- The medication list is stored only in Streamlit session state and is not persisted. Reminder notifications are not implemented.
- API usage may incur charges or be subject to provider limits. Review your provider's current terms.

## Troubleshooting
- **Gemini not configured:** confirm `.env` is in the same folder as `app.py`, the variable is `GEMINI_API_KEY`, and restart Streamlit.
- **API error:** verify the key, account access, quota, model availability and internet connection.
- **Package import error:** activate the virtual environment and run `pip install -r requirements.txt` again.
- **Port already in use:** run `streamlit run app.py --server.port 8502`.
- **No evidence retrieved:** try a more specific query matching the topics in the knowledge base. The app should disclose missing evidence rather than guess.

## Included sources
Source URLs are listed in the JSON knowledge base. The excerpts are concise demo notes, not a replacement for the full source. Re-check source availability and content before public deployment.
