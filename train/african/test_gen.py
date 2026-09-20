import json, random, re
from collections import Counter
from build_full_corpus import check_quality_gates

# Test generator with rich variation
OPENINGS = [
    "To control {pest} effectively in {town},",
    "{pest} management in {town} relies on",
    "Farmers across {reg} identify {pest} by",
    "Field scouting in {town} during {season} reveals",
    "When {pest} appears in {reg},",
    "Scout maize fields in {town} twice weekly for",
    "In {reg}, controlling {pest} requires",
    "Early intervention against {pest} in {town} involves",
    "Recognize {pest} infestation in {reg} through",
    "Extension recommendations from {inst} in {town} highlight",
    "Biological management of {pest} in {reg} uses",
    "Cultural control of {pest} in {town} begins with",
    "Inspect 20 plants per location in {town} for",
    "Yield protection in {reg} against {pest} starts with",
    "Smallholders in {town} manage {pest} by",
    "Scouting data from {inst} in {reg} shows",
    "Preventing {pest} damage in {town} requires",
    "Apply dry wood ash or sand in {town} when",
    "Integrated management of {pest} in {reg} combines",
    "During {season} in {town}, protect crops from"
]

Q_FRAMES = [
    "What is the most effective approach for a farmer in {town}, {country} to manage {pest}?",
    "How should an extension officer in {reg} guide farmers dealing with {pest} during {season}?",
    "A smallholder in {town} notices {symp}. What causes this and what action is needed?",
    "In {reg}, what cultural practices reduce {pest} damage on {crop} without high costs?",
    "What scouting procedure should a lead farmer in {town} follow for {pest}?",
    "Why is early detection of {pest} critical for growers in {reg} during {season}?",
    "How does crop rotation with legumes help farmers in {town} suppress {pest}?",
    "What advice does {inst} provide for smallholders in {reg} facing {pest} outbreaks?",
    "Can a farmer in {town} use biological controls against {pest}? How?",
    "What steps protect harvested {crop} from {pest} in {reg} storage facilities?"
]

print("Test generator script ready.")
