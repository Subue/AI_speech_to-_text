from __future__ import annotations

from datetime import datetime


def execute_intent(intent: str, user_text: str) -> dict:
    intent = intent.lower().strip()

    if intent == "greeting":
        return {"intent": intent, "reply": "Hello. How can I help you today?"}
    if intent == "goodbye":
        return {"intent": intent, "reply": "Goodbye. Have a great day."}
    if intent == "thanks":
        return {"intent": intent, "reply": "You are welcome."}
    if intent == "time_query":
        current_time = datetime.now().strftime("%I:%M %p")
        return {"intent": intent, "reply": f"The current time is {current_time}."}
    if intent == "date_query":
        current_date = datetime.now().strftime("%d %B %Y")
        return {"intent": intent, "reply": f"Today's date is {current_date}."}
    if intent == "project_info":
        return {
            "intent": intent,
            "reply": (
                "This is an ML-powered voice assistant project with a trainable "
                "intent classifier, FastAPI backend, and browser dashboard."
            ),
        }
    if intent == "capabilities":
        return {
            "intent": intent,
            "reply": (
                "I can classify commands, answer greetings, tell date and time, "
                "describe the project, and retrain the model from the dataset."
            ),
        }
    if intent == "status_check":
        return {"intent": intent, "reply": "All core services are working correctly."}

    return {
        "intent": "unknown",
        "reply": (
            "I am not fully confident about that command yet. "
            "Please expand the dataset with more examples and retrain the model."
        ),
    }
