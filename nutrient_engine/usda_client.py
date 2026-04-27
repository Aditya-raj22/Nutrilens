"""USDA FoodData Central API wrapper."""
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.nal.usda.gov/fdc/v1"


class USDAClient:
    def __init__(self, api_key=None, timeout=15):
        self.api_key = api_key or os.environ.get("USDA_API_KEY")
        if not self.api_key:
            raise ValueError("Set USDA_API_KEY env var")
        self.timeout = timeout

    PREFERRED_DATA_TYPES = ("Foundation", "SR Legacy", "Survey (FNDDS)")

    def search(self, query, page_size=5, data_type=PREFERRED_DATA_TYPES):
        params = {"query": query, "pageSize": page_size, "api_key": self.api_key}
        try:
            r = requests.get(f"{BASE}/foods/search", params=params, timeout=self.timeout)
            r.raise_for_status()
        except requests.HTTPError as e:
            raise requests.HTTPError(f"USDA search failed: {e.response.status_code}") from None
        foods = r.json().get("foods", [])
        if data_type:
            allowed = set(data_type)
            filtered = [f for f in foods if f.get("dataType") in allowed]
            if filtered:
                foods = filtered
        return foods

    def food(self, fdc_id):
        try:
            r = requests.get(
                f"{BASE}/food/{fdc_id}",
                params={"api_key": self.api_key},
                timeout=self.timeout,
            )
            r.raise_for_status()
        except requests.HTTPError as e:
            raise requests.HTTPError(f"USDA food({fdc_id}) failed: {e.response.status_code}") from None
        return r.json()

    def nutrient_profile(self, fdc_id):
        """Return {nutrient_name: value_per_100g} for a given FDC food id."""
        data = self.food(fdc_id)
        out = {}
        for n in data.get("foodNutrients", []):
            name = n.get("nutrient", {}).get("name") or n.get("nutrientName")
            val = n.get("amount") if "amount" in n else n.get("value")
            if name and val is not None:
                out[name] = val
        return out

    def throttle_search(self, query, **kw):
        """Throttled search for batch mapping build (USDA limits 1000/hr)."""
        result = self.search(query, **kw)
        time.sleep(0.5)
        return result
