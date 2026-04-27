"""OpenAI API wrapper for the recommendation layer."""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from .prompts import PROMPTS


class OpenAIClient:
    def __init__(self, api_key=None, model=None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5")

    def recommend(self, deficiencies, user_profile, variant="few_shot_cot",
                  max_tokens=4000, reasoning_effort="low"):
        system = PROMPTS[variant]
        user = f"User profile: {json.dumps(user_profile)}\nDeficiencies: {json.dumps(deficiencies)}"
        is_gpt5 = self.model.startswith("gpt-5")
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # GPT-5 reasoning models: max_completion_tokens covers reasoning + output.
        # Our CoT is expressed in the prompt, so "low" internal reasoning is enough.
        if is_gpt5:
            kwargs["max_completion_tokens"] = max_tokens
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["max_tokens"] = max_tokens
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
