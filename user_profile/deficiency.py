"""Flag deficient nutrients with gap magnitude."""


def flag_deficiencies(consumed: dict, rda: dict, threshold=0.8):
    gaps = {}
    for nutrient, target in rda.items():
        got = consumed.get(nutrient, 0.0)
        if target <= 0:
            continue
        ratio = got / target
        if ratio < threshold:
            gaps[nutrient] = {
                "consumed": round(got, 2),
                "rda": round(target, 2),
                "pct_met": round(ratio * 100, 1),
                "gap": round(target - got, 2),
            }
    return gaps
