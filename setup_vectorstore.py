# agent/setup_vectorstore.py  (free local version)
# Uses sentence-transformers for embeddings — no API key needed
# Run: python agent/setup_vectorstore.py

import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Same sample docs as before ────────────────
SAMPLE_DOCS = [
    {
        "source": "pump_maintenance_manual.txt",
        "content": """
Centrifugal Pump Maintenance Manual — Section 4: Failure Modes

4.1 Seal Failures
Mechanical seal failure is the most common cause of centrifugal pump downtime,
accounting for approximately 70% of all pump failures. Signs include:
- Visible leakage around the seal housing
- Increased vibration (typically >4.5 mm/s RMS)
- Rising bearing temperatures above 80°C
- Unusual noise during operation

Root causes: dry running, incorrect installation, misalignment >0.05mm,
excessive shaft deflection, abrasive particles in the process fluid.

4.2 Cavitation
Cavitation occurs when local fluid pressure drops below vapour pressure,
forming and collapsing vapour bubbles. Indicators:
- Crackling or rattling noise (similar to gravel in the pump)
- Pitting on impeller surfaces
- Reduction in flow rate and head pressure
- Vibration increase at 1x and 2x running frequency

Prevention: maintain NPSH available > NPSH required by at least 0.5m.
Common fix: raise suction tank level or reduce suction pipe losses.

4.3 Bearing Failures
Rolling element bearings typically last 20,000-50,000 hours under normal
conditions. Premature failure causes:
- Contamination (most common — use ISO 4406 cleanliness targets)
- Overlubrication (causes churning heat)
- Misalignment (induces axial loads beyond bearing rating)
- Electrical fluting (requires insulated bearings or shaft grounding)
"""
    },
    {
        "source": "pump_p101_incident_log.txt",
        "content": """
Equipment ID: P-101 (Centrifugal Process Pump)
Location: Unit 3 — Cooling Water Circuit
Period: January 2024

2024-01-03: Vibration alarm at 08:42. Peak 5.2 mm/s RMS on NDE bearing.
Operator reduced load 15%. Cause: suction strainer 40% blocked.
Action: strainer cleaned, alarm cleared.

2024-01-09: Bearing temperature high alarm. DE bearing reached 87°C (normal 55-70°C).
Pump isolated at 14:30. Cause: over-greasing — grease migrated into housing.
Action: cleaned, regreased with 25g SKF LGMT3. Returned to service 2024-01-10.

2024-01-15: Cavitation noise reported. Flow dropped 85 to 61 m³/h.
NPSH available dropped to 2.1m against required 2.8m.
Cause: upstream valve partially closed during maintenance.
Action: valve fully opened, flow restored in 10 minutes.

2024-01-22: No incidents. Vibration 2.8 mm/s (normal <4.5). Temperatures normal.

Summary: 3 incidents, all resolved within same shift. No parts replaced.
Recommendation: install differential pressure gauge across suction strainer.
"""
    },
    {
        "source": "vibration_analysis_guidelines.txt",
        "content": """
Vibration Analysis Guidelines — ISO 10816-3 Severity Zones (Pumps >15kW):
Zone A (0-2.3 mm/s RMS):   New equipment, good condition
Zone B (2.3-4.5 mm/s RMS): Acceptable for long-term operation
Zone C (4.5-7.1 mm/s RMS): Unsatisfactory — investigate within 2 weeks
Zone D (>7.1 mm/s RMS):    Dangerous — shut down immediately

Frequency Diagnosis:
1x RPM:            Unbalance, misalignment, bent shaft
2x RPM:            Angular misalignment, looseness
Blade pass freq:   Impeller damage, cavitation
Sub-synchronous:   Fluid instability, oil whirl
High frequency:    Bearing defect frequencies

Bearing Defect Frequencies:
BPFO (outer race): 3.0-3.5 x RPM
BPFI (inner race): 4.5-5.5 x RPM
BSF (ball spin):   1.8-2.2 x RPM
FTF (cage):        0.35-0.45 x RPM

Zone C action: Spectrum analysis, check alignment, schedule maintenance within 14 days.
Zone D action: Isolate immediately, mandatory inspection, replace bearings minimum.
"""
    },
    {
        "source": "predictive_maintenance_schedule.txt",
        "content": """
Predictive Maintenance Schedule — Rotating Equipment

Monthly: Vibration survey all bearing housings, bearing temperature checks,
visual inspection (leaks, noise, coupling guard), lubrication log check.

Quarterly: Vibration FFT spectrum analysis critical pumps, alignment check
pumps >22kW, strainer differential pressure review, motor current analysis >30kW.

Annual: Full mechanical inspection (impeller wear, wear rings, shaft runout),
bearing replacement >25,000 hours, seal inspection (replace if >2 drops/min),
performance test vs design curve.

Critical pumps P-101, P-103, P-201: continuous online vibration monitoring,
4-hourly bearing temperature rounds, 2-hour alarm response time.

Lubrication — P-101:
Grease: SKF LGMT3 (mineral oil, lithium complex, NLGI 3)
DE bearing:  25g every 2000 running hours
NDE bearing: 20g every 2000 running hours
Never mix grease types — purge and repack if changing type.
"""
    }
]


def setup_vectorstore():
    print("Setting up ChromaDB with local sentence-transformer embeddings...")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads ~90MB — subsequent runs use cache)\n")

    # Local embeddings — no API key
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", " "]
    )

    documents = []
    for item in SAMPLE_DOCS:
        chunks = splitter.split_text(item["content"])
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                documents.append(Document(
                    page_content=chunk.strip(),
                    metadata={"source": item["source"], "chunk": i}
                ))

    print(f"Created {len(documents)} chunks from {len(SAMPLE_DOCS)} documents")

    # Delete existing collection to avoid conflicts on re-run
    import shutil
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        print("Cleared existing chroma_db")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="equipment_docs"
    )

    count = vectorstore._collection.count()
    print(f"\n✓ ChromaDB ready at ./chroma_db")
    print(f"✓ {count} vectors stored")
    print(f"\nNext step: python agent/agent.py")
    return vectorstore


if __name__ == "__main__":
    setup_vectorstore()