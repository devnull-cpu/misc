from ollama import chat
from pydantic import BaseModel, Field
from pathlib import Path
from PIL import Image
import argparse
import tempfile
import json


# --- Schemas ---

class FootballMatchInfo(BaseModel):
    home_team: str = Field(description="Name of the home team as shown on screen, keep abbreviated if so")
    away_team: str = Field(description="Name of the away team as shown on screen, keep abbreviated if so")
    home_score: int | None = Field(description="Goals scored by the home team, null if not visible")
    away_score: int | None = Field(description="Goals scored by the away team, null if not visible")
    match_time: str | None = Field(description="Current match time or status, e.g. '45:00', null if not visible")


class TennisMatchInfo(BaseModel):
    side_1: list[str] = Field(description="Player name(s) on side 1 as shown on screen, keep abbreviated if so. If doubles, split into separate entries")
    side_2: list[str] = Field(description="Player name(s) on side 2 as shown on screen, keep abbreviated if so. If doubles, split into separate entries")
    s1_scores: list[int] = Field(description="All score numbers shown for side 1, left to right, e.g. [0, 6, 5]")
    s2_scores: list[int] = Field(description="All score numbers shown for side 2, left to right, e.g. [1, 6, 6]")
    
# --- Prompts and examples ---

SPORTS = {
    "football": {
        "schema": FootballMatchInfo,
        "example": {
            "home_team": "LPL",
            "away_team": "MAN",
            "home_score": 2,
            "away_score": 1,
            "match_time": "78:32",
        },
        "prompt": (
            "Extract the match information from this image. Look at the score overlay.\n\n"
            "Return a JSON object with these fields:\n"
            "- home_team: name of the home team, keep as is if abbreviated (null if not visible)\n"
            "- away_team: name of the away team, keep as is if abbreviated (null if not visible)\n"
            "- home_score: goals scored by home team (null if not visible)\n"
            "- away_score: goals scored by away team (null if not visible)\n"
            "- match_time: current time or status like '45:00' (null if not visible)\n\n"
        ),
    },
    "tennis": {
        "schema": TennisMatchInfo,
        "example": {
            "side_1": ["Bolelli", "Vavassori"],
            "side_2": ["Arevalo", "Pavic"],
            "s1_scores": [1, 3, 30],
            "s2_scores": [0, 2, 40]
        },
        "prompt": (
            "Extract the tennis match information from this image. Look at the score overlay.\n\n"
            "Score bugs show player names on the left, then several columns of numbers to the right."
            "A '<' or '>' or dot/circle next to a player indicates they are serving.\n\n"
            "Return a JSON object with these fields:\n"
            "- side_1: player name(s) on side 1 as shown, keep abbreviated if so. If doubles, split into separate entries\n"
            "- side_2: player name(s) on side 2 as shown, keep abbreviated if so. If doubles, split into separate entries\n"
            "- s1_scores: all numbers shown for side 1, read left to right as they appear on screen\n"
            "- s2_scores: all numbers shown for side 2, read left to right as they appear on screen\n\n"
        ),
    },
}


def ensure_compatible_image(image_path: str) -> str:
    if Path(image_path).suffix.lower() in (".webp", ".bmp", ".tiff"):
        img = Image.open(image_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        return tmp.name
    return image_path


def extract(image_path: str, sport: str, model: str):
    config = SPORTS[sport]
    schema = config["schema"]
    image_path = ensure_compatible_image(image_path)

    prompt = config["prompt"] + f"Example output:\n{json.dumps(config['example'], indent=2)}"
    
    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path],
            }
        ],
        format=schema.model_json_schema(),
        options={"temperature": 0},
    )

    return schema.model_validate_json(response.message.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract score bug info from broadcast images")
    parser.add_argument("image", help="Path to the image")
    parser.add_argument("-s", "--sport", choices=SPORTS.keys(), default="football", help="Sport type (default: football)")
    parser.add_argument("-m", "--model", default="qwen3-vl:8b-instruct-q8_0", help="Ollama model to use")
    args = parser.parse_args()

    result = extract(args.image, args.sport, args.model)
    print(json.dumps(result.model_dump(), indent=2))