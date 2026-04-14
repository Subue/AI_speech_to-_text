const commandInput = document.getElementById("commandInput");
const predictionResult = document.getElementById("predictionResult");
const trainingStatus = document.getElementById("trainingStatus");
const metricsPanel = document.getElementById("metricsPanel");

async function fetchMetrics() {
  metricsPanel.textContent = "Loading metrics...";
  const response = await fetch("/api/model-info");
  const data = await response.json();
  metricsPanel.innerHTML = `
Dataset size: <span class="metric-value">${data.dataset_size}</span>
Intent classes: <span class="metric-value">${data.intent_count}</span>
Accuracy: <span class="metric-value">${(data.accuracy * 100).toFixed(2)}%</span>
Train split: ${data.train_size}
Test split: ${data.test_size}
Intents: ${data.intents.join(", ")}
  `.trim();
}

async function predictIntent() {
  const text = commandInput.value.trim();
  if (!text) {
    predictionResult.textContent = "Enter a command before prediction.";
    return;
  }

  predictionResult.textContent = "Predicting...";
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await response.json();

  predictionResult.textContent = [
    `Intent: ${data.prediction.intent}`,
    `Confidence: ${(data.prediction.confidence * 100).toFixed(2)}%`,
    `Assistant Reply: ${data.action.reply}`,
  ].join("\n");
}

async function trainModel() {
  trainingStatus.textContent = "Training model...";
  const response = await fetch("/api/train", { method: "POST" });
  const data = await response.json();
  if (!response.ok) {
    trainingStatus.textContent = data.detail || "Training failed.";
    return;
  }
  trainingStatus.textContent = `Training complete.\nAccuracy: ${(data.accuracy * 100).toFixed(2)}%\nSamples: ${data.dataset_size}`;
  fetchMetrics();
}

function activateBrowserSpeech() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    predictionResult.textContent = "Browser speech recognition is not supported here.";
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.start();
  predictionResult.textContent = "Listening from browser microphone...";

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    commandInput.value = transcript;
    predictIntent();
  };

  recognition.onerror = () => {
    predictionResult.textContent = "Browser voice recognition failed.";
  };
}

document.getElementById("predictButton").addEventListener("click", predictIntent);
document.getElementById("trainButton").addEventListener("click", trainModel);
document.getElementById("refreshButton").addEventListener("click", fetchMetrics);
document.getElementById("voiceButton").addEventListener("click", activateBrowserSpeech);

fetchMetrics();
