
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

load_dotenv()
ROOT = Path(__file__).parent
KB_PATH = ROOT / "knowledge_base" / "trusted_medical_sources.json"

st.set_page_config(page_title="MediPlate AI", page_icon="🥗", layout="wide", initial_sidebar_state="expanded")

# ---------- Design ----------
st.markdown("""
<style>
:root { --mp-ink:#eaf1f7; --mp-muted:#aab9c8; --mp-teal:#4dd6bd; }
.stApp { background: radial-gradient(ellipse at top left, #173342 0%, #0b1722 42%, #08111a 100%); color:var(--mp-ink); }
[data-testid="stSidebar"] { background: #101f2b; border-right: 1px solid #263b49; }
.block-container {
    padding-top: 3rem;
    max-width: 1320px;
}
.hero { padding: 1.45rem 1.6rem; border:1px solid #294b58; border-radius:22px; background:linear-gradient(120deg,rgba(23,62,72,.92),rgba(19,35,51,.88)); margin-bottom:1.1rem; }
.eyebrow { color:#73e2cb; font-weight:700; letter-spacing:.14em; font-size:.72rem; text-transform:uppercase; }
.hero h1 { color:#f4fbff; font-size:2.45rem; line-height:1.05; margin:.4rem 0 .6rem; }
.hero p { color:#b9cbd5; max-width:760px; margin:0; }
.kpi { padding:1rem; border-radius:16px; background:rgba(21,43,57,.82); border:1px solid #2b4655; min-height:110px; }
.kpi .label { color:#aabfca; font-size:.82rem; }
.kpi .value { color:#f3fbff; font-size:1.35rem; font-weight:750; margin-top:.35rem; }
.panel { padding:1rem 1.15rem; border-radius:16px; background:rgba(15,31,43,.85); border:1px solid #294150; }
.small-muted { color:#aab9c8; font-size:.86rem; }
.stButton > button { border-radius:12px; font-weight:700; }
div[data-testid="stAlert"] { border-radius:12px; }
</style>
<div class="hero">
  <div class="eyebrow">PERSONALIZED HEALTH • EVIDENCE FIRST</div>
  <h1>MediPlate <span style="color:#73e2cb">AI</span></h1>
  <p>Your food, medication and nutrition companion. See what the system knows, where its evidence comes from, and when a clinician should verify a result.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Knowledge retrieval ----------
@st.cache_data
def load_knowledge():
    if not KB_PATH.exists():
        return []
    try:
        data = json.loads(KB_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def retrieve(query: str, docs: List[Dict[str, Any]], top_k: int = 4):
    if not query.strip() or not docs:
        return []
    texts = [f"{d.get('title','')} {d.get('topic','')} {d.get('content','')}" for d in docs]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [{"doc": docs[i], "score": float(score)} for i, score in ranked[:top_k] if score > 0.01]
    except ValueError:
        return []

def source_block(results):
    if not results:
        return "No matching local evidence was retrieved. Do not infer a medical conclusion."
    parts = []
    for i, result in enumerate(results, 1):
        d = result["doc"]
        parts.append(f"[S{i}] {d.get('title')} | {d.get('publisher')} | {d.get('url')}\nTopic: {d.get('topic')}\nEvidence excerpt: {d.get('content')}")
    return "\n\n".join(parts)

def get_client():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or genai is None:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None

def call_gemini(prompt: str, image: Image.Image = None):
    client = get_client()
    if client is None:
        return None, "Gemini is not configured. Add GEMINI_API_KEY to your .env file and restart the app."
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    try:
        contents = [prompt]
        if image is not None:
            contents.append(image)
        response = client.models.generate_content(model=model, contents=contents)
        return getattr(response, "text", None) or "The model returned no text.", None
    except Exception as exc:
        return None, f"Gemini request failed: {exc}"

def clean_json(text: str):
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None

# ---------- Sidebar profile ----------
st.sidebar.markdown("## Your demo profile")
st.sidebar.caption("Use fictional data only. This prototype is not for real medical decisions.")
profile_name = st.sidebar.text_input("Profile label", "Demo Patient")
conditions = st.sidebar.multiselect(
    "Health conditions",
    ["Diabetes", "High blood pressure", "Kidney disease", "High cholesterol", "Food allergy", "None"],
    default=["Diabetes"]
)
allergies = st.sidebar.text_input("Known allergies", "peanuts")
medicines = st.sidebar.text_area("Current medicines (name only)", "Example: warfarin", height=85,
                                  help="Use a fictional example for the demo. Never enter a real patient's identifiable information.")
diet = st.sidebar.selectbox("Diet preference", ["No preference", "Vegetarian", "Vegan", "Halal", "Lactose-free", "Gluten-free"])
other = st.sidebar.text_area("Other dietary restrictions / notes", "", height=70)
profile = f"Conditions: {', '.join(conditions) or 'not provided'}. Allergies: {allergies or 'not provided'}. Medicines: {medicines or 'not provided'}. Diet: {diet}. Notes: {other or 'none'}."

docs = load_knowledge()
with st.sidebar.expander("RAG status", expanded=False):
    st.write(f"Local knowledge records: **{len(docs)}**")
    st.write("Retrieval method: **TF-IDF lexical retrieval**")
    st.write("Model: **Gemini multimodal**" if get_client() else "Model: **not configured**")
    st.caption("This prototype retrieves excerpts from its bundled knowledge base. It does not browse the web live.")

tab_scan, tab_ask, tab_plan, tab_meds, tab_about = st.tabs(
    ["📷 Food scanner", "🔎 Ask with evidence", "🥗 Meal ideas", "⏰ Medication schedule", "🧪 How it works"]
)

# ---------- Food scanner ----------
with tab_scan:
    st.subheader("Scan a meal")
    st.write("Upload a meal photo. The AI will suggest likely foods, then assess only what can be supported by the profile and retrieved evidence.")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png", "webp"], key="food_image")
        if uploaded:
            try:
                food_image = Image.open(uploaded).convert("RGB")
                st.image(food_image, caption="Uploaded meal photo", use_container_width=True)
            except Exception:
                food_image = None
                st.error("Could not read that image. Try a JPG or PNG.")
        else:
            food_image = None
            st.markdown('<div class="panel"><div style="font-size:2rem">🥗</div><b>Demo tip</b><p class="small-muted">Choose a clear photo with the full meal visible. A photo cannot reliably reveal hidden ingredients, cross-contamination, freshness, or exact nutrition.</p></div>', unsafe_allow_html=True)
    with right:
        meal_context = st.text_area("Anything you know about the meal?", placeholder="Example: home-cooked rice, lentil curry and yogurt; no packaged label available.", height=100)
        confirmed = st.text_input("Ingredients you can confirm (optional)", placeholder="e.g. rice, lentils, yogurt, salt")
        if st.button("Analyze meal safely", type="primary", use_container_width=True):
            if not food_image:
                st.warning("Upload a meal photo first.")
            else:
                query = f"Food: {meal_context}. Confirmed ingredients: {confirmed}. Patient profile: {profile}. Check food suitability, allergy risks, condition-specific nutrition and drug-food interactions. "
                matches = retrieve(query, docs, 5)
                evidence = source_block(matches)
                prompt = f"""
You are MediPlate AI, a cautious health-information assistant for a COLLEGE DEMO, not a clinician.
Analyze the supplied food image and user context. Do not claim certainty from an image. Hidden ingredients, cross-contact, freshness, portion size and exact nutrition cannot be reliably inferred from pixels.
Patient profile (fictional/demo): {profile}
User's meal notes: {meal_context or 'none'}
User-confirmed ingredients: {confirmed or 'none'}
Retrieved local evidence (use only when relevant; cite IDs exactly):
{evidence}

Return valid JSON only with keys:
food_identification (string),
identified_items (array of strings),
uncertainties (array of strings),
classification (one of "Suitable", "Caution", "Avoid/Verify"),
reasoning (string),
patient_specific_checks (array of strings),
safer_alternatives (array of strings),
professional_follow_up (string),
sources (array of objects with source_id, title, url, relevance).
Safety rules:
- Use "Avoid/Verify" if a stated allergy may be present or a serious risk cannot be ruled out; ask the user to verify rather than assert the food contains it.
- Never invent a drug-food interaction. If the local evidence does not cover the medicine or combination, say it was not verified and recommend pharmacist/clinician confirmation.
- Do not advise starting/stopping/changing medicine or dosage.
- Do not diagnose or calculate exact nutrient values from a photo.
- If evidence is weak, say so and choose Caution or Avoid/Verify, not a confident green light.
- Use sources only from retrieved evidence; if none support a claim, say no supporting local source was retrieved.
"""
                with st.spinner("Inspecting image and retrieving relevant evidence…"):
                    answer, err = call_gemini(prompt, food_image)
                if err:
                    st.error(err)
                    st.info("You can still demonstrate RAG retrieval in the evidence panel below, but image analysis needs a working Gemini API key.")
                else:
                    parsed = clean_json(answer)
                    st.session_state["last_scan"] = {"answer": parsed, "raw": answer, "matches": matches}
    if "last_scan" in st.session_state:
        scan = st.session_state["last_scan"]
        parsed = scan.get("answer")
        with right:
            st.markdown("---")
            st.subheader("Assessment")
            if parsed:
                classification = parsed.get("classification", "Caution")
                if classification == "Suitable":
                    st.success("Assessment: Suitable — within the limits of available evidence")
                elif classification == "Avoid/Verify":
                    st.error("Assessment: Avoid / Verify")
                else:
                    st.warning("Assessment: Caution")
                st.markdown(f"**Likely identification:** {parsed.get('food_identification','Not confidently identified')}")
                st.markdown(f"**Why:** {parsed.get('reasoning','No explanation returned.')}")
                for heading, key in [("What needs checking", "patient_specific_checks"), ("Uncertainties", "uncertainties"), ("Possible alternatives", "safer_alternatives")]:
                    items = parsed.get(key, [])
                    if items:
                        st.markdown(f"**{heading}**")
                        for item in items:
                            st.write(f"• {item}")
                if parsed.get("professional_follow_up"):
                    st.info(parsed["professional_follow_up"])
            else:
                st.warning("The model response was not valid JSON. Raw response is shown for debugging.")
                st.code(scan.get("raw", ""), language="text")
        with st.expander("Evidence retrieved by RAG", expanded=True):
            if scan.get("matches"):
                for idx, match in enumerate(scan["matches"], 1):
                    d = match["doc"]
                    st.markdown(f"**[S{idx}] {d.get('title')}** — {d.get('publisher')} · relevance score `{match['score']:.3f}`")
                    st.write(d.get("content"))
                    st.markdown(f"[Open source]({d.get('url')})")
            else:
                st.warning("No sufficiently matching local source was retrieved. The system should not make a strong medical claim.")
        st.caption("Prototype result only. Not a diagnosis or a guarantee of safety. Confirm ingredients and significant risks with a qualified professional.")

# ---------- Evidence Q&A ----------
with tab_ask:
    st.subheader("Ask a health question")
    st.write("This tab makes the retrieval-augmented generation (RAG) process visible: query → retrieved passages → grounded answer.")
    question = st.text_area("Question", placeholder="Example: What should I verify before eating a meal when I have a peanut allergy?", height=95)
    if st.button("Retrieve evidence and answer", type="primary"):
        matches = retrieve(question + " " + profile, docs, 5)
        st.markdown("#### Retrieved evidence")
        if matches:
            for i, item in enumerate(matches, 1):
                d = item["doc"]
                st.markdown(f"**[S{i}] {d.get('title')}** · score `{item['score']:.3f}`")
                st.write(d.get("content"))
                st.markdown(f"[Source: {d.get('publisher')}]({d.get('url')})")
        else:
            st.warning("No relevant evidence was found in the bundled knowledge base. The app should not guess.")
        if question.strip():
            prompt = f"""Answer as a cautious health information assistant, not a clinician.
Patient context: {profile}
Question: {question}
Retrieved evidence:
{source_block(matches)}
Use the retrieved evidence as the basis of the answer. Cite sources as [S1], [S2], etc. If evidence does not answer the question, explicitly say so and suggest a pharmacist/clinician or official source. Do not diagnose, change medicines or invent interactions. Keep the answer concise."""
            with st.spinner("Generating a grounded answer…"):
                answer, err = call_gemini(prompt)
            st.markdown("#### Answer")
            if err:
                st.error(err)
                st.info("Retrieval still ran locally; connect Gemini to generate the answer.")
            else:
                st.write(answer)

# ---------- Meal ideas ----------
with tab_plan:
    st.subheader("Personalized meal ideas")
    st.write("Generate ideas with caveats based on the current demo profile and retrieved nutrition evidence.")
    meal_goal = st.selectbox("What are you planning?", ["Everyday balanced meals", "Lower-sodium ideas", "Diabetes-aware meal ideas", "Vegetarian meals", "Allergy-aware meal ideas", "High-Protien meals", "Heart-healthy meals", "Kidney-friendly meal ideas", "Weight-management meals", "Indian meal ideas",],)
    avoid = st.text_input("Ingredients to avoid", allergies)
    if st.button("Generate meal ideas", type="primary"):
        query = f"{meal_goal}. Patient profile: {profile}. Avoid: {avoid}. Nutrition and condition-specific food advice."
        matches = retrieve(query, docs, 5)
        prompt = f"""Create 3 cautious meal ideas for a demo user. Profile: {profile}. Goal: {meal_goal}. Ingredients to avoid: {avoid}.
Retrieved evidence:
{source_block(matches)}
For each idea include meal, why it may fit, checks/uncertainties, and which source IDs support it. Do not promise medical benefit, calculate exact nutrition, or treat generic ideas as a prescribed diet. If the sources are insufficient for the condition, say so and advise a registered dietitian/clinician."""
        with st.spinner("Preparing evidence-aware meal ideas…"):
            answer, err = call_gemini(prompt)
        if err:
            st.error(err)
        else:
            st.markdown(answer)
        with st.expander("Retrieved sources"):
            for i, m in enumerate(matches, 1):
                d = m["doc"]
                st.markdown(f"**[S{i}] {d.get('title')}** — [{d.get('publisher')}]({d.get('url')})")
                st.write(d.get("content"))

# ---------- Medication schedule ----------
with tab_meds:
    st.subheader("Medication schedule builder")
    st.warning("This is a reminder organizer only. Enter the schedule exactly as prescribed; it does not determine doses or replace medical instructions.")
    with st.form("med_form"):
        med_name = st.text_input("Medicine name", placeholder="As written on prescription")
        med_instructions = st.text_input("Prescribed instructions", placeholder="Copy exactly from verified prescription")
        med_time = st.time_input("Reminder time")
        med_relation = st.selectbox("Food timing (only if prescribed/verified)", ["Not specified", "Before food", "With food", "After food"])
        med_submit = st.form_submit_button("Add reminder")
    if med_submit:
        if not med_name.strip() or not med_instructions.strip():
            st.error("Enter the medicine name and the exact prescribed instructions.")
        else:
            st.session_state.setdefault("med_schedule", []).append({
                "name": med_name.strip(), "instructions": med_instructions.strip(),
                "time": med_time.strftime("%I:%M %p"), "food": med_relation
            })
            st.success("Added to this session's reminder list.")
    if st.session_state.get("med_schedule"):
        st.markdown("#### Today's reminder list")
        for i, item in enumerate(sorted(st.session_state["med_schedule"], key=lambda x: x["time"])):
            col1, col2, col3 = st.columns([1, 2, 1])
            col1.markdown(f"**{item['time']}**")
            col2.markdown(f"**{item['name']}**  \n{item['instructions']}  \nFood timing: {item['food']}")
            taken = col3.checkbox("Taken", key=f"taken_{i}")
        st.caption("Reminders exist only while this app session is running; persistent storage and push notifications are not implemented in this prototype.")

# ---------- Methodology ----------
with tab_about:
    st.subheader("How the prototype works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="kpi"><div class="label">01 · INPUT</div><div class="value">Patient context</div><p class="small-muted">Conditions, allergies, diet and medicine names.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi"><div class="label">02 · RETRIEVAL</div><div class="value">Local RAG</div><p class="small-muted">TF-IDF retrieves relevant trusted excerpts before generation.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi"><div class="label">03 · OUTPUT</div><div class="value">Grounded response</div><p class="small-muted">Gemini interprets evidence and explains uncertainty.</p></div>', unsafe_allow_html=True)
    st.markdown("#### Safety and limitations")
    st.markdown("""
- A food photo cannot reliably identify every ingredient, allergen, contaminant, spoilage state or portion size.
- Local retrieval can miss relevant information. A missing result does **not** mean no interaction exists.
- Medicine names can be misspelled or ambiguous. Confirm medicine details with a pharmacist or prescriber.
- Do not use this prototype for emergency decisions, diagnosis, medication dosing, or changes to prescribed treatment.
- The demo profile is entered in the browser session; do not use real identifiable patient data.
""")
    st.markdown("#### Current implementation")
    st.write(f"Knowledge records loaded: {len(docs)}")
    st.write("Retriever: TF-IDF with unigrams and bigrams.")
    st.write("Image reasoning and text generation: Gemini API (when configured).")
    st.write("Storage: session-only; no database or persistent patient record storage.")
    st.markdown("#### Test checklist")
    st.markdown("""
1. Ask a question that clearly matches one knowledge record and verify the correct source is retrieved.
2. Ask an unrelated question and verify the app says evidence is insufficient.
3. Upload a clear food photo and confirm the model lists uncertainty.
4. Add a known allergen to the profile and check the result asks for verification instead of guaranteeing safety.
5. Test the app with no API key and confirm retrieval and reminder UI still work.
""")

st.markdown("---")
st.markdown('<p class="small-muted">MediPlate AI · Evidence-informed, uncertainty-aware, not a medical device.</p>', unsafe_allow_html=True)
