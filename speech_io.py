from __future__ import annotations


def speak_text(text: str) -> None:
    print(f"Assistant: {text}")
    try:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


def voice_input_available() -> bool:
    try:
        import speech_recognition as sr  # type: ignore

        with sr.Microphone():
            return True
    except Exception:
        return False


def listen_for_command() -> str:
    try:
        import speech_recognition as sr  # type: ignore
    except Exception:
        print("Speech recognition is not available.")
        return ""

    try:
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        recognizer.non_speaking_duration = 0.5
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=12)

        text = recognizer.recognize_google(audio).strip()
        print(f"You: {text}")
        return text
    except sr.WaitTimeoutError:
        return ""
    except sr.UnknownValueError:
        return ""
    except (sr.RequestError, OSError):
        print("Voice input is unavailable.")
       
        return ""



