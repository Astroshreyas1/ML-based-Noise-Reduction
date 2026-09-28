"""System-1 typed decisions with Laya (non-autoregressive: choice / score / yes-no in one forward pass,
~35 ms, nothing generated so nothing hallucinated). Base checkpoint now; `finetune/` makes a radio-
procedure checkpoint from labelled transcripts (feature: "finetune laya for this usecase").
"""
from __future__ import annotations

QUESTIONS = {
    "msg_type": {"type": "choice", "instructions": "What kind of military radio transmission is this?",
                 "criteria": {"contact_report": "enemy seen, contact, taking fire, engaging",
                              "medevac": "casualties, wounded, medevac, nine line",
                              "fire_mission": "call for fire, adjust fire, fire for effect, splash",
                              "check_fire": "check fire, cease fire",
                              "sitrep": "situation report, status, ammo, no contact",
                              "radio_check": "radio check, how do you read, loud and clear",
                              "movement": "moving, en route, rally point, phase line, ETA",
                              "other": "anything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this transmission for the commander?",
                "criteria": ["routine", "priority", "immediate"]},
    "enemy_contact": {"type": "noul", "instructions": "Does the sender report enemy contact or taking fire?"},
    "casualties": {"type": "noul", "instructions": "Does the sender report friendly casualties?"},
}


def load_laya(device: str = "cuda", checkpoint: str | None = None):
    import laya
    if checkpoint:
        return laya.load(checkpoint, device=device)
    return laya.load("convaiinnovations/laya", device=device)


def decide(model, text: str) -> dict:
    r = model.predict(text, QUESTIONS) if hasattr(model, "predict") else model(text, QUESTIONS)
    a = r["answers"]
    return {"msg_type": a["msg_type"]["choice"], "msg_type_conf": float(a["msg_type"]["confidence"]),
            "urgency": float(a["urgency"]["score"]), "enemy_contact": float(a["enemy_contact"]["noul"]),
            "casualties": float(a["casualties"]["noul"])}
