import psycopg
import ollama
from groq import Groq
import os
import base64
from datetime import datetime

# ------------------------------
# Variables
# ------------------------------
EMBED_MODEL = "embeddinggemma"
LLM_MODEL = "llama3"
VISION_MODEL = "llama3.2-vision"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = "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" 

db_connection_str = "dbname=rag_chatbot user=postgres password=1803 host=localhost port=5432"
TOP_K = 5

groq_client = Groq(api_key=GROQ_API_KEY)

# Ensure output folder exists
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# ------------------------------
# Helper functions
# ------------------------------

def calculate_embeddings(corpus: str) -> list[float]:
    response = ollama.embeddings(EMBED_MODEL, corpus)
    return response["embedding"]

def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(str(v) for v in vec) + "]"

def retrieve_chunks(query: str, k: int = TOP_K):
    embedding = calculate_embeddings(query)
    pg_vec = to_pgvector(embedding)

    with psycopg.connect(db_connection_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT corpus, embedding <=> %s::vector AS distance
                FROM embeddings
                ORDER BY distance ASC
                LIMIT %s
                """,
                (pg_vec, k)
            )
            results = cur.fetchall()

    return [r[0] for r in results]

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ------------------------------
# AGENT 1: Vision Agent (FIXED)
# ------------------------------

def vision_agent(image_path: str) -> dict:
    print("   👁️  Agent 1 — Analyzing visual features...")

    try:
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image not found: {image_path}"}

        print(f"   📂 File: {os.path.basename(image_path)}")
        img = encode_image_to_base64(image_path)

        response = ollama.chat(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": """Analyze this medical brain image and describe SPECIFIC VISUAL FEATURES you observe:

1. **Location**: Which brain region/hemisphere shows abnormalities?
2. **Appearance**: Describe the shape, size, borders, density/intensity
3. **Texture**: Is it uniform, heterogeneous, with necrosis, edema?
4. **Surrounding tissue**: Any mass effect, midline shift, edema, compression?
5. **Signal characteristics**: Dark/bright areas, contrast enhancement patterns

Be SPECIFIC and DETAILED about what you visually observe. Focus on measurable, objective features.
Do NOT diagnose - only describe visual characteristics.""",
                    "images": [img]
                }
            ]
        )

        desc = response["message"]["content"].strip()
        return {"success": True, "description": desc}

    except Exception as e:
        return {"success": False, "error": str(e)}

# ------------------------------
# AGENT 2: Textbook Retrieval (IMPROVED)
# ------------------------------

def textbook_retrieval_agent(vision_description: str, top_k: int = TOP_K) -> dict:
    print("   🔎 Agent 2 — Retrieving relevant textbook passages...")

    try:
        # Create a better search query by extracting key medical terms
        query_parts = [
            "brain tumor",
            "glioblastoma",
            "mass lesion",
            vision_description[:600]  # Include visual description
        ]
        
        search_query = " ".join(query_parts)
        chunks = retrieve_chunks(search_query, top_k)

        if not chunks:
            return {"success": False, "error": "No textbook passages retrieved.", "retrieved_chunks": []}

        return {"success": True, "retrieved_chunks": chunks, "num_chunks": len(chunks)}

    except Exception as e:
        return {"success": False, "error": str(e), "retrieved_chunks": []}

# ------------------------------
# AGENT 3: Enrichment (FIXED)
# ------------------------------

def enrichment_agent(chunks: list[str], vision_description: str) -> dict:
    print("   🧠 Agent 3 — Synthesizing analysis with medical knowledge...")

    try:
        joined = "\n\n═════════════════\n\n".join(chunks)
        if len(joined) > 6000:
            joined = joined[:6000] + "\n\n[Additional content truncated]"

        system_msg = {
            "role": "system",
            "content": """You are a medical education AI assistant. Your role is to:
1. Analyze the visual findings from the brain scan
2. Connect these findings with relevant medical knowledge from textbooks
3. Explain what these findings typically indicate in medical literature
4. Provide educational context about the condition

You must:
- Be specific about the observed features
- Reference medical knowledge to explain significance
- Provide educational information about typical presentations
- Suggest what further clinical evaluation might involve
- End with a clear disclaimer

You must NOT:
- Make definitive diagnoses
- Provide treatment recommendations
- Replace professional medical evaluation"""
        }

        user_msg = {
            "role": "user",
            "content": f"""**VISUAL ANALYSIS FROM BRAIN SCAN:**

{vision_description}

**RELEVANT MEDICAL TEXTBOOK KNOWLEDGE:**

{joined}

**YOUR TASK:**

Based on the visual features described above and the medical knowledge provided, create a comprehensive educational analysis that:

1. **Summarizes the key visual findings** - What specific features are observed?

2. **Medical context** - Based on the textbook knowledge, what do these visual characteristics typically indicate in medical literature? What conditions commonly present with these features?

3. **Clinical significance** - Explain the importance of these findings (edema, mass effect, etc.)

4. **Typical clinical workflow** - What additional imaging or tests are typically performed for such presentations?

5. **Educational summary** - Synthesize the information into a clear educational explanation

**End with this exact disclaimer:**

⚠️ DISCLAIMER: This analysis is for educational purposes only and does not constitute medical advice, diagnosis, or treatment recommendation. Actual diagnosis requires comprehensive clinical evaluation by qualified healthcare professionals, including detailed patient history, physical examination, and correlation with clinical symptoms. If you have medical concerns, please consult a qualified healthcare provider."""
        }

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[system_msg, user_msg],
            max_tokens=2000,
            temperature=0.3
        )

        final = response.choices[0].message.content.strip()
        return {"success": True, "final_text": final}

    except Exception as e:
        return {"success": False, "error": str(e)}

# ------------------------------
# PIPELINE
# ------------------------------

def analyze_pipeline(image_path: str) -> dict:
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║         AI MEDICAL IMAGE ANALYSIS PIPELINE                    ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")

    # Agent 1
    print("┌─── STAGE 1: VISUAL ANALYSIS ───────────────────────────────────┐")
    v = vision_agent(image_path)
    if not v["success"]:
        return {"success": False, "error": v["error"]}
    vision_desc = v["description"]
    print("│ ✓ Visual features extracted")
    print("└────────────────────────────────────────────────────────────────┘\n")

    # Agent 2
    print("┌─── STAGE 2: KNOWLEDGE RETRIEVAL ───────────────────────────────┐")
    t = textbook_retrieval_agent(vision_desc, TOP_K)
    if not t["success"]:
        return {"success": False, "error": t["error"]}
    chunks = t["retrieved_chunks"]
    print(f"│ ✓ Retrieved {t['num_chunks']} relevant passages from medical literature")
    print("└────────────────────────────────────────────────────────────────┘\n")

    # Agent 3
    print("┌─── STAGE 3: SYNTHESIS & ENRICHMENT ────────────────────────────┐")
    e = enrichment_agent(chunks, vision_desc)
    if not e["success"]:
        return {"success": False, "error": e["error"]}
    print("│ ✓ Analysis synthesized with medical knowledge")
    print("└────────────────────────────────────────────────────────────────┘\n")

    return {
        "success": True,
        "vision_analysis": vision_desc,
        "retrieved_chunks": chunks,
        "final_synthesis": e["final_text"],
        "sources": t["num_chunks"]
    }

# ------------------------------
# CLI
# ------------------------------

def main():
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║       MEDICAL IMAGE ANALYSIS - MULTI-AGENT SYSTEM             ║")
    print("║                                                               ║")
    print("║  Commands:                                                    ║")
    print("║    analyze <image_path>  - Analyze a brain scan image        ║")
    print("║    exit/quit            - Exit the program                   ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")

    while True:
        cmd = input("💬 Command: ").strip()

        if cmd.lower() in ("exit", "quit"):
            print("\n👋 Exiting system. Goodbye!\n")
            break

        if cmd.startswith("analyze "):
            image_path = cmd.split(" ", 1)[1].strip()

            print(f"\n🔍 Starting analysis for: {os.path.basename(image_path)}\n")
            result = analyze_pipeline(image_path)

            if not result["success"]:
                print(f"\n❌ ERROR: {result['error']}\n")
                continue

            # BUILD FILE NAME
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_path = f"outputs/analysis_{timestamp}.txt"

            # WRITE COMPLETE REPORT
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("═══════════════════════════════════════════════════════════════\n")
                f.write("           MEDICAL IMAGE ANALYSIS REPORT\n")
                f.write("═══════════════════════════════════════════════════════════════\n\n")
                f.write(f"Image: {os.path.basename(image_path)}\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Sources Referenced: {result['sources']}\n\n")
                f.write("───────────────────────────────────────────────────────────────\n")
                f.write("VISUAL ANALYSIS (Agent 1)\n")
                f.write("───────────────────────────────────────────────────────────────\n\n")
                f.write(result["vision_analysis"])
                f.write("\n\n───────────────────────────────────────────────────────────────\n")
                f.write("SYNTHESIZED ANALYSIS (Agent 3)\n")
                f.write("───────────────────────────────────────────────────────────────\n\n")
                f.write(result["final_synthesis"])
                f.write("\n\n═══════════════════════════════════════════════════════════════\n")

            print("\n╔═══════════════════════════════════════════════════════════════╗")
            print("║                  ANALYSIS COMPLETE                            ║")
            print("╚═══════════════════════════════════════════════════════════════╝\n")
            print("📊 VISUAL FEATURES (Preview):\n")
            print(result["vision_analysis"][:350] + "...\n")
            print(f"📚 Medical Sources: {result['sources']} passages retrieved\n")
            print("💾 Full Report Saved To:")
            print(f"   📄 {out_path}\n")
            print("═══════════════════════════════════════════════════════════════\n")

            input("Press ENTER to continue...")
            print("\n")

        else:
            print("❌ Unknown command. Use: analyze <path> or exit\n")

if __name__ == "__main__":
    main()