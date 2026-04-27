"""Food-101 class name → USDA FDC ID mapping builder and loader."""
import json
from pathlib import Path

MAPPING_PATH = Path("data/food101_fdc_map.json")


def build_mapping(client, class_names, out_path=MAPPING_PATH):
    mapping = {}
    for i, name in enumerate(class_names):
        query = name.replace("_", " ")
        try:
            results = client.throttle_search(query, page_size=3)
        except Exception as e:
            print(f"[{i+1}/{len(class_names)}] {name}: ERROR {e}")
            mapping[name] = None
            continue
        if not results:
            print(f"[{i+1}/{len(class_names)}] {name}: NO MATCH")
            mapping[name] = None
            continue
        top = results[0]
        mapping[name] = {
            "fdc_id": top["fdcId"],
            "description": top.get("description"),
            "data_type": top.get("dataType"),
        }
        print(f"[{i+1}/{len(class_names)}] {name} -> {top.get('description')}")
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(json.dumps(mapping, indent=2))
    return mapping


def load_mapping(path=MAPPING_PATH):
    return json.loads(Path(path).read_text())
