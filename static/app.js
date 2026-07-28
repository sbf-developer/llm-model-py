const statusEl = document.getElementById("status");
const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const trainBtn = document.getElementById("train-btn");
const reloadBtn = document.getElementById("reload-btn");
const trainLog = document.getElementById("train-log");

let history = "";
let pollTimer = null;

function setTab(name) {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${name}-panel`);
  });
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
    data.model_loaded ? "model loaded" : "model not loaded",
    data.training ? "training..." : "idle",
  ];
  statusEl.textContent = parts.join(" · ");
  trainBtn.disabled = data.training;
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
  }
}

trainBtn.addEventListener("click", async () => {
  trainLog.textContent = "starting...";
  const res = await fetch("/api/train/start", { method: "POST" });
  if (!res.ok) {
    const err = await res.json();
    trainLog.textContent = err.detail || "Failed to start training";
    return;
  }
  refreshStatus();
  if (!pollTimer) pollTrainLogs();
});

reloadBtn.addEventListener("click", async () => {
  const res = await fetch("/api/model/reload", { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || "Reload failed");
    return;
  }
  refreshStatus();
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

messagesEl.innerHTML = '<div class="empty">Train a model first, then chat here.<br>Replies will be rough — that\'s expected.</div>';
refreshStatus();
setInterval(refreshStatus, 5000);
