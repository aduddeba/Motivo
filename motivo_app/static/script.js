import "dotenv/config";

const apiKey = process.env.OPENAI_API_KEY;

let currentQ = 0;

async function loadQuestion() {
  const res = await fetch(`/get_question/${currentQ}`);
  const data = await res.json();

  if (data.end) {
    // Send to prediction
    const predictRes = await fetch("/predict", { method: "POST" });
    const result = await predictRes.json();
    window.location.href = `/result?car=${encodeURIComponent(
      result.dream_car
    )}`;
    return;
  }

  document.getElementById("question-text").textContent = data.question;
  const answerArea = document.getElementById("answer-area");
  answerArea.innerHTML = "";

  if (data.type === "number") {
    answerArea.innerHTML = `<input id="answer-input" type="number" />`;
  } else if (data.type === "select") {
    let options = data.options
      .map((opt) => `<option value="${opt}">${opt}</option>`)
      .join("");
    answerArea.innerHTML = `<select id="answer-input">${options}</select>`;
  }
}

async function nextQuestion() {
  const answer = document.getElementById("answer-input").value;
  const res = await fetch("/get_question/" + currentQ);
  const data = await res.json();

  if (!data.end) {
    await fetch("/submit_answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: data.key, answer }),
    });
  }

  currentQ++;
  loadQuestion();
}

window.onload = loadQuestion;
