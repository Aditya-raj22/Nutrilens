"""Compare the three OpenAI prompt variants on a fixed deficiency profile.

Records output length, structural validity (JSON parse), and qualitative score
(manually reviewed in the accompanying markdown).
"""
import json
from pathlib import Path

from llm_client.openai_client import OpenAIClient
from llm_client.prompts import PROMPTS


FIXTURE = {
    "user_profile": {
        "weight_kg": 65, "height_cm": 170, "age": 25, "sex": "female",
        "activity": "moderate",
    },
    "deficiencies": {
        "Iron, Fe": {"consumed": 9.9, "rda": 18, "pct_met": 55.0, "gap": 8.1},
        "Vitamin D (D2 + D3)": {"consumed": 6.0, "rda": 15, "pct_met": 40.0, "gap": 9.0},
    },
}


def try_json(text):
    s = text.find("[")
    e = text.rfind("]")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def main():
    client = OpenAIClient()
    out = Path("eval/outputs/prompt_comparison.json")
    out.parent.mkdir(exist_ok=True, parents=True)
    results = {}
    for name in PROMPTS:
        print(f"running {name}...")
        resp = client.recommend(
            FIXTURE["deficiencies"], FIXTURE["user_profile"], variant=name,
        )
        parsed = try_json(resp)
        results[name] = {
            "raw": resp,
            "parsed": parsed,
            "valid_json": parsed is not None,
            "n_recs": len(parsed) if parsed else 0,
            "char_count": len(resp),
        }
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    print("\nSummary:")
    for name, r in results.items():
        print(f"  {name}: valid_json={r['valid_json']} n_recs={r['n_recs']} chars={r['char_count']}")


if __name__ == "__main__":
    main()
