import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

# (question, pdf file name, page where the answer is)
tests = [
    # ---- Budget speech ----
    ("What is the fiscal deficit target for 2026-27?", "budget_2026-2027.pdf", 23),
    ("What are the gross market borrowings estimated for 2026-27?", "budget_2026-2027.pdf", 23),
    ("What public capex is proposed for FY2026-27?", "budget_2026-2027.pdf", 11),
    ("What is the outlay for Biopharma SHAKTI?", "budget_2026-2027.pdf", 8),
    ("What is the proposed STT rate on futures?", "budget_2026-2027.pdf", 29),
    ("What common safe harbour margin is proposed for IT services?", "budget_2026-2027.pdf", 27),
    ("Till which year is the tax holiday for foreign cloud companies using Indian data centres?", "budget_2026-2027.pdf", 28),
    ("What vertical share of devolution to States was retained?", "budget_2026-2027.pdf", 22),
    # ---- RBI annual report ----
    ("What did the MPC decide on the repo rate in April 2026?", "RBI annual report 2025-2026.pdf", 35),
    ("What were India's foreign exchange reserves at end-March 2026?", "RBI annual report 2025-2026.pdf", 31),
    ("What is the projected CPI inflation for 2026-27?", "RBI annual report 2025-2026.pdf", 35),
    ("What is the projected real GDP growth for 2026-27?", "RBI annual report 2025-2026.pdf", 35),
    ("How many UPI transactions were recorded and how much did volume grow?", "RBI annual report 2025-2026.pdf", 32),
    ("How much surplus was payable to the Central Government for 2025-26?", "RBI annual report 2025-2026.pdf", 196),
    # ---- Financial Stability Report ----
    ("What was the GNPA ratio of large borrowers at end-March 2026?", "RBI financial stability report_2026.pdf", 83),
    ("How fast did SCBs' credit grow during 2025-26?", "RBI financial stability report 2026.pdf", 79),
    ("What is the CET1 ratio of the 46 select banks in March 2026?", "RBI financial stability report2026.pdf", 88),
    ("What is the aggregate GNPA ratio of 46 banks in March 2026?", "RBI financial stability report2026.pdf", 89),
    ("What was the all-time high monthly SIP inflow in March 2026?", "RBI financial stability report2026.pdf", 48),
    ("How much would system CRAR fall if equity prices drop 25, 35 and 55 per cent?", "RBI financial stability report2026.pdf", 95),
]
         

def norm(s):
    s = s.lower()
    if "financial" in s:
        return "fsr"
    if "annual" in s:
        return "ar"
    return "budget"

hits4 = hits10 = 0
for q, src, page in tests:
    res = col.query(query_embeddings=model.encode([q]).tolist(), n_results=10)
    got = [(norm(m["source"]), m["page"]) for m in res["metadatas"][0]]
    want = (norm(src), page)
    in4 = want in got[:4]
    in10 = want in got 
    hits4 += in4
    hits10 += in10
    print("OK4" if in4 else ("OK10" if in10 else "MISS"), q)
    if not in10:
        print("      got:", [(s[:12], p) for s, p in got[:4]], "want:", (want[0][:12], page))
print(f"hit@4: {hits4}/{len(tests)} = {hits4/len(tests):.0%}")
print(f"hit@10: {hits10}/{len(tests)} = {hits10/len(tests):.0%}")