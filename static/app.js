const statusEl = document.getElementById("status");
const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const trainResumeBtn = document.getElementById("train-resume-btn");
const trainFreshBtn = document.getElementById("train-fresh-btn");
const reloadBtn = document.getElementById("reload-btn");
const trainLog = document.getElementById("train-log");
const checkpointList = document.getElementById("checkpoint-list");
const dataStats = document.getElementById("data-stats");
const dataUser = document.getElementById("data-user");
const dataAssistant = document.getElementById("data-assistant");
const dataText = document.getElementById("data-text");
const dataDialogueBtn = document.getElementById("data-dialogue-btn");
const dataTextBtn = document.getElementById("data-text-btn");
const dataMsg = document.getElementById("data-msg");

let history = "";
let pollTimer = null;

function setTab(name) {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${name}-panel`);
  });
  if (name === "data") refreshDataStats();
  if (name === "train") refreshCheckpoints();
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => setTab(btn.dataset.tab));
});

function addMessage(role, text) {
  const empty = messagesEl.querySelector(".empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function refreshStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  const parts = [
    data.device,
    data.checkpoint_exists ? "checkpoint ready" : "no checkpoint",
    data.training ? "training..." : "idle",
  ];
  statusEl.textContent = parts.join(" · ");
  const busy = data.training;
  trainResumeBtn.disabled = busy;
  trainFreshBtn.disabled = busy;
}

async function refreshCheckpoints() {
  const res = await fetch("/api/checkpoints");
  const data = await res.json();
  if (!data.checkpoints.length) {
    checkpointList.textContent = "No checkpoints yet.";
    return;
  }
  checkpointList.innerHTML = data.checkpoints
    .map(
      (c) =>
        `<div class="ckpt-row"><span>${c.name}</span><span>step ${c.step}${
          c.val_loss != null ? ` · loss ${Number(c.val_loss).toFixed(3)}` : ""
        }</span></div>`
    )
    .join("");
}

async function refreshDataStats() {
  const res = await fetch("/api/data/stats");
  const d = await res.json();
  dataStats.textContent = d.exists
    ? `${d.chars.toLocaleString()} characters · ${d.lines.toLocaleString()} lines`
    : "No data file yet.";
}

async function pollTrainLogs() {
  const res = await fetch("/api/train/logs");
  const data = await res.json();
  trainLog.textContent = data.logs.join("\n") || "No logs yet.";
  trainLog.scrollTop = trainLog.scrollHeight;
  if (data.training) {
    pollTimer = setTimeout(pollTrainLogs, 1000);
  } else {
    pollTimer = null;
    refreshStatus();
    refreshCheckpoints();
  }
}

async function startTraining(fresh) {
  trainLog.textContent = "starting...";
  const res = await fetch("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fresh }),
  });
  if (!res.ok) {
    const err = await res.json();
    trainLog.textContent = err.detail || "Failed to start training";
    return;
  }
  refreshStatus();
  if (!pollTimer) pollTrainLogs();
}

trainResumeBtn.addEventListener("click", () => startTraining(false));
trainFreshBtn.addEventListener("click", () => {
  if (confirm("Fresh start ignores saved checkpoints. Continue?")) startTraining(true);
});

reloadBtn.addEventListener("click", async () => {
  const res = await fetch("/api/model/reload", { method: "POST" });
  const data = await res.json();
  if (!res.ok) alert(data.detail || "Reload failed");
  refreshStatus();
});

dataDialogueBtn.addEventListener("click", async () => {
  dataMsg.textContent = "";
  const res = await fetch("/api/data/dialogue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user: dataUser.value, assistant: dataAssistant.value }),
  });
  const d = await res.json();
  if (!res.ok) {
    dataMsg.textContent = d.detail || "Failed";
    return;
  }
  dataUser.value = "";
  dataAssistant.value = "";
  dataMsg.textContent = `Added. Total: ${d.chars.toLocaleString()} chars. Run train to learn it.`;
  refreshDataStats();
});

dataTextBtn.addEventListener("click", async () => {
  dataMsg.textContent = "";
  const res = await fetch("/api/data/append", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: dataText.value }),
  });
  const d = await res.json();
  if (!res.ok) {
    dataMsg.textContent = d.detail || "Failed";
    return;
  }
  dataText.value = "";
  dataMsg.textContent = `Added ${d.appended_chars.toLocaleString()} chars. Total: ${d.chars.toLocaleString()}. Run train to learn it.`;
  refreshDataStats();
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  addMessage("user", message);
  chatInput.value = "";
  chatInput.disabled = true;
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    });
    const data = await res.json();
    if (!res.ok) {
      addMessage("bot", data.detail || "Something went wrong.");
      return;
    }
    history = data.history;
    addMessage("bot", data.reply);
  } catch {
    addMessage("bot", "Network error.");
  } finally {
    chatInput.disabled = false;
    chatInput.focus();
  }
});

messagesEl.innerHTML =
  '<div class="empty">Train a model first, then chat here.<br>Replies will be rough — that\'s expected.</div>';
refreshStatus();
setInterval(refreshStatus, 5000);
