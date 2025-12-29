from llm.llm_factory import get_llm
from analysis.prompts import generate_prompts

def run_analysis(topic, persona, models, num_prompts):
    results = {}

    for model_name in models:
        model_key = model_name.lower().strip()  # ⭐ FIX

        llm = get_llm(model_key)

        prompts = generate_prompts(
            topic=topic,
            persona=persona,
            num=num_prompts,
            llm=llm
        )

        results[model_key] = {
            "prompts": prompts
        }

    return results
