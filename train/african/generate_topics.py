import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent.resolve()

COUNTIES = [
    "Nakuru", "Kitale", "Trans Nzoia", "Eldoret", "Uasin Gishu", "Meru", "Embu", "Machakos",
    "Bungoma", "Kisumu", "Kakamega", "Nyeri", "Kirinyaga", "Murang'a", "Makueni", "Kilifi",
    "Kwale", "Arusha", "Mbeya", "Morogoro", "Iringa", "Dodoma", "Mwanza", "Kilimanjaro",
    "Shinyanga", "Tabora", "Tanga", "Ruvuma", "Kampala", "Mbale", "Gulu", "Masaka",
    "Jinja", "Mbarara", "Fort Portal", "Musanze", "Huye", "Kigali", "Bugesera", "Oromia", "Amhara"
]

SEASONS = [
    "the long rains (Masika)", "the short rains (Vuli)", "the early long rains",
    "the late short rains", "the dry period after harvest", "the Belg season", "the Meher long rains"
]

QUESTIONERS = [
    "A smallholder farmer in {county}",
    "An agricultural extension officer visiting a farm in {county}",
    "A young farmer starting out in {county}",
    "A member of a farmers' cooperative in {county}",
    "An agrodealer advising a client in {county}",
    "A farmer managing 2 acres in {county}",
    "A community extension agent working near {county}"
]

# Helper to format clean text
def clean_txt(t):
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)
    return t.strip()

print("Generator helper loaded.")
