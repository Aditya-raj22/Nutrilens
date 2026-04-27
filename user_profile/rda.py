"""Personal RDA: Harris-Benedict BMR/TDEE + DRI lookup tables."""


ACTIVITY = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}

# Revised Harris-Benedict (Roza & Shizgal, 1984)
def bmr(weight_kg, height_cm, age, sex):
    if sex == "male":
        return 88.362 + 13.397 * weight_kg + 4.799 * height_cm - 5.677 * age
    return 447.593 + 9.247 * weight_kg + 3.098 * height_cm - 4.330 * age


def tdee(weight_kg, height_cm, age, sex, activity="moderate"):
    return bmr(weight_kg, height_cm, age, sex) * ACTIVITY[activity]


# Simplified adult DRIs (19-50). Units match USDA nutrient names where possible.
DRI = {
    "male": {
        "Iron, Fe": 8,                 # mg
        "Calcium, Ca": 1000,           # mg
        "Vitamin C, total ascorbic acid": 90,  # mg
        "Vitamin D (D2 + D3)": 15,     # µg
        "Zinc, Zn": 11,                # mg
        "Potassium, K": 3400,          # mg
        "Magnesium, Mg": 400,          # mg
        "Fiber, total dietary": 38,    # g
    },
    "female": {
        "Iron, Fe": 18,
        "Calcium, Ca": 1000,
        "Vitamin C, total ascorbic acid": 75,
        "Vitamin D (D2 + D3)": 15,
        "Zinc, Zn": 8,
        "Potassium, K": 2600,
        "Magnesium, Mg": 310,
        "Fiber, total dietary": 25,
    },
}


def personal_rda(weight_kg, height_cm, age, sex, activity="moderate"):
    rda = dict(DRI[sex])
    rda["Protein"] = 0.8 * weight_kg                             # g/kg
    rda["Energy"] = tdee(weight_kg, height_cm, age, sex, activity)  # kcal
    return rda
