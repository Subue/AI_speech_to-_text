from __future__ import annotations

from actions import execute_intent
from ml_intents import IntentClassifier, ensure_model_artifacts
from speech_io import listen_for_command, speak_text, voice_input_available


EXIT_PHRASES = {"exit", "quit", "stop", "bye"}


def main() -> None:
    ensure_model_artifacts()
    classifier = IntentClassifier.load()

    if not voice_input_available():
        speak_text("Voice input is not available. Switching to text mode.")
        while True:
            text = input("You: ").strip()
            if not text:
                continue
            if text.lower() in EXIT_PHRASES:
                speak_text("Goodbye.")
                break
            prediction = classifier.predict(text)
            result = execute_intent(prediction.intent, text)
            speak_text(result["reply"])
        return

    speak_text("Assistant is ready. Speak your command.")
    while True:
        command = listen_for_command()
        if not command:
            continue
        if command.lower().strip() in EXIT_PHRASES:
            speak_text("Goodbye.")
            break
        prediction = classifier.predict(command)
        result = execute_intent(prediction.intent, command)
        speak_text(result["reply"])


if __name__ == "__main__":
    main()
