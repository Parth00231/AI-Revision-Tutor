"use strict";

// ============================================================
// CONFIG & BASE URL
// ============================================================
const API_BASE = window.location.protocol === "file:"
  ? "http://localhost:8001"
  : "";

// ============================================================
// APPLICATION STATE
// ============================================================
let isBusy = false;
let allTopics = [];
let activeModalSource = null;

// ============================================================
// INITIALIZATION
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  // Configure marked for smooth revision rendering
  if (window.marked) {
    marked.setOptions({
      breaks: true,
      gfm: true,
    });
  }

  checkStatus();
  loadTopics();
  document.getElementById("chat-input")?.focus();
});

// ============================================================
// BACKEND STATUS HEALTH CHECK
// ============================================================
async function checkStatus() {
  const dot  = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  if (!dot || !text) return;

  try {
    const res  = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("Health check returned status " + res.status);

    const data = await res.json();
    if (data.collection_ready) {
      dot.className = "status-dot";
      const pts = data.points_count ? ` (${data.points_count} chunks)` : "";
      text.textContent = `Ready · Qdrant Active${pts}`;
    } else {
      dot.className = "status-dot loading";
      text.textContent = "Collection missing · Run embed";
    }
  } catch (err) {
    dot.className = "status-dot offline";
    text.textContent = "Server offline";
    showToast("Cannot connect to server. Check server.py.", "error");
  }
}

// ============================================================
// LECTURE TOPICS LIST & FILTER
// ============================================================
async function loadTopics() {
  const list  = document.getElementById("topic-list");
  const badge = document.getElementById("topic-count");
  if (!list) return;

  try {
    const res  = await fetch(`${API_BASE}/topics`);
    if (!res.ok) throw new Error("Failed to load topics");
    const data = await res.json();

    allTopics = data.topics || [];
    if (badge) badge.textContent = allTopics.length;
    renderTopicList(allTopics);
  } catch (err) {
    list.innerHTML = `<li style="padding:10px 12px;font-size:0.75rem;color:var(--red)">Failed to load lectures</li>`;
    if (badge) badge.textContent = "!";
  }
}

function renderTopicList(topics) {
  const list = document.getElementById("topic-list");
  if (!list) return;

  list.innerHTML = "";
  if (!topics.length) {
    list.innerHTML = `<li style="padding:12px;font-size:0.75rem;color:var(--text-3);text-align:center">No matching lectures</li>`;
    return;
  }

  topics.forEach((topic) => {
    // Extract video number and title from pattern e.g. "1. Introduction to AI.txt"
    const match = topic.match(/^(\d+)\.\s*(.+?)(?:\s*\.txt)?$/i);
    const num   = match ? match[1] : "•";
    const label = match ? match[2].trim() : topic.replace(/\.txt$/i, "").trim();

    const li = document.createElement("li");
    li.className     = "topic-item";
    li.title         = `Click to revise: ${label}`;
    li.dataset.topic = label;
    li.dataset.raw   = topic;

    li.innerHTML = `
      <span class="topic-num">${num}</span>
      <span class="topic-name">${esc(label)}</span>
    `;

    li.addEventListener("click", () => {
      document.querySelectorAll(".topic-item").forEach(el => el.classList.remove("active"));
      li.classList.add("active");

      // Switch to notes and populate input
      switchTab("notes");
      const inp = document.getElementById("notes-input");
      if (inp) {
        inp.value = label;
        inp.focus();
      }
    });

    list.appendChild(li);
  });
}

function filterTopics(query) {
  const q = query.trim().toLowerCase();
  if (!q) {
    renderTopicList(allTopics);
    return;
  }
  const filtered = allTopics.filter(t => t.toLowerCase().includes(q));
  renderTopicList(filtered);
}

// ============================================================
// TAB NAVIGATION
// ============================================================
function switchTab(tab) {
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  const navBtn = document.getElementById(`nav-${tab}`);
  if (navBtn) navBtn.classList.add("active");

  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  const panel = document.getElementById(`panel-${tab}`);
  if (panel) panel.classList.add("active");

  if (tab === "ask") {
    document.getElementById("chat-input")?.focus();
  } else if (tab === "notes") {
    document.getElementById("notes-input")?.focus();
  }
}

// ============================================================
// ASK TAB — RAG Q&A
// ============================================================
function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 140) + "px";
}

function handleChatKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
}

function askChip(btn) {
  const input = document.getElementById("chat-input");
  if (input) {
    input.value = btn.textContent.trim();
    sendQuestion();
  }
}

async function sendQuestion() {
  if (isBusy) return;
  const input = document.getElementById("chat-input");
  const question = input.value.trim();
  if (!question) return;

  isBusy = true;
  input.value = "";
  input.style.height = "auto";
  const sendBtn = document.getElementById("send-btn");
  if (sendBtn) sendBtn.disabled = true;

  // Remove welcome card if present
  const welcome = document.getElementById("welcome-card");
  if (welcome) welcome.remove();

  // Render user question
  appendMsg("user", question);

  // Add typing placeholder
  const typingId = addTyping();

  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 5 }),
    });

    removeTyping(typingId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      appendMsg("ai", `⚠️ ${err.detail || "Server error occurred while answering."}`, []);
      showToast("Question failed", "error");
    } else {
      const data = await res.json();
      appendMsg("ai", data.answer, data.sources || []);
    }
  } catch (err) {
    removeTyping(typingId);
    appendMsg("ai", "⚠️ Could not reach the server. Make sure `python server.py` is running.", []);
    showToast("Connection failed", "error");
  } finally {
    isBusy = false;
    if (sendBtn) sendBtn.disabled = false;
    document.getElementById("chat-input")?.focus();
  }
}

function appendMsg(role, content, sources = []) {
  const area = document.getElementById("chat-messages");
  if (!area) return;

  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;

  const renderedContent = role === "ai"
    ? `<div class="md">${marked.parse(content)}</div>`
    : `<div>${esc(content)}</div>`;

  const copyAction = role === "ai"
    ? `<button class="msg-copy-btn" onclick="copyText(this, ${JSON.stringify(content)})" title="Copy answer">📋 Copy</button>`
    : "";

  msg.innerHTML = `
    <div class="msg-bubble">${renderedContent}</div>
    <div class="msg-meta">
      <span>${role === "user" ? "You" : "AI Tutor"} · ${time}</span>
      ${copyAction}
    </div>
  `;

  // Render clickable source chips for AI answers
  if (role === "ai" && sources.length) {
    const row = document.createElement("div");
    row.className = "source-row";
    sources.slice(0, 5).forEach((s) => {
      const label = cleanName(s.source);
      const chip = document.createElement("div");
      chip.className = "src-chip";
      chip.title = "Click to view lecture transcript excerpt";
      chip.innerHTML = `🎬 ${esc(label)} <span class="src-score">${s.score}</span>`;
      chip.addEventListener("click", () => openSourceModal(s));
      row.appendChild(chip);
    });
    msg.appendChild(row);
  }

  area.appendChild(msg);
  area.scrollTop = area.scrollHeight;
}

function addTyping() {
  const area = document.getElementById("chat-messages");
  const id = "typing-" + Date.now();
  const el = document.createElement("div");
  el.id = id;
  el.className = "typing-row";
  el.innerHTML = `
    <div class="typing-bubble">
      <div class="t-dot"></div>
      <div class="t-dot"></div>
      <div class="t-dot"></div>
    </div>
  `;
  area.appendChild(el);
  area.scrollTop = area.scrollHeight;
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function clearChat() {
  const area = document.getElementById("chat-messages");
  if (!area) return;

  area.innerHTML = `
    <div class="welcome-card" id="welcome-card">
      <div class="welcome-emoji">🎓</div>
      <h2 class="welcome-title">What would you like to revise today?</h2>
      <p class="welcome-sub">Ask any concept from all 23 course lectures. Answers are grounded in the lecture transcripts with citations.</p>
      <div class="chips-grid">
        <button class="chip" onclick="askChip(this)">What is RAG and how does it work?</button>
        <button class="chip" onclick="askChip(this)">How do embeddings work?</button>
        <button class="chip" onclick="askChip(this)">Explain the transformer attention mechanism</button>
        <button class="chip" onclick="askChip(this)">What are chunking strategies?</button>
        <button class="chip" onclick="askChip(this)">What is prompt engineering and few-shot prompting?</button>
        <button class="chip" onclick="askChip(this)">How does Qdrant perform cosine similarity search?</button>
      </div>
    </div>
  `;
  document.getElementById("chat-input")?.focus();
}

// ============================================================
// STUDY NOTES GENERATION
// ============================================================
function handleNotesKey(e) {
  if (e.key === "Enter") {
    e.preventDefault();
    generateNotes();
  }
}

async function generateNotes() {
  if (isBusy) return;
  const input = document.getElementById("notes-input");
  const topic = input?.value.trim();
  if (!topic) {
    showToast("Please enter a topic or lecture title", "info");
    input?.focus();
    return;
  }

  isBusy = true;
  const btn = document.getElementById("notes-btn");
  const btnIcon = document.getElementById("notes-btn-icon");
  const btnLabel = document.getElementById("notes-btn-label");
  if (btn) btn.disabled = true;
  if (btnIcon) btnIcon.textContent = "⏳";
  if (btnLabel) btnLabel.textContent = "Generating…";

  const output = document.getElementById("notes-output");
  const emptyEl = document.getElementById("notes-empty");
  if (emptyEl) emptyEl.style.display = "none";

  const shimId = addShimmer(output);

  try {
    const res = await fetch(`${API_BASE}/notes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic, top_k: 7 }),
    });

    removeShimmer(shimId, output);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showToast(`Generation failed: ${err.detail || "Server error"}`, "error");
      if (emptyEl) emptyEl.style.display = "";
    } else {
      const data = await res.json();
      renderNotesCard(output, data);
      showToast("Notes generated successfully!", "success");
    }
  } catch (err) {
    removeShimmer(shimId, output);
    showToast("Failed to communicate with server.", "error");
    if (emptyEl) emptyEl.style.display = "";
  } finally {
    isBusy = false;
    if (btn) btn.disabled = false;
    if (btnIcon) btnIcon.textContent = "✨";
    if (btnLabel) btnLabel.textContent = "Generate Notes";
  }
}

function addShimmer(container) {
  const id = "shim-" + Date.now();
  const el = document.createElement("div");
  el.id = id;
  el.className = "notes-shimmer";
  el.innerHTML = `
    <div class="sh-line sh-h"></div>
    <div class="sh-line" style="width:90%"></div>
    <div class="sh-line" style="width:80%"></div>
    <div class="sh-line" style="width:65%"></div>
    <div class="sh-line sh-h" style="margin-top:10px"></div>
    <div class="sh-line" style="width:85%"></div>
    <div class="sh-line" style="width:75%"></div>
    <div class="sh-line" style="width:95%"></div>
  `;
  container.prepend(el);
  return id;
}

function removeShimmer(id, container) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function renderNotesCard(container, data) {
  const card = document.createElement("div");
  card.className = "notes-card";
  card.dataset.rawText = data.notes || "";
  card.dataset.topic = data.topic || "Study Notes";

  const sourcesHtml = (data.sources || []).slice(0, 6).map((s) => {
    const label = cleanName(s.source);
    return `<div class="src-chip" title="Click to view transcript excerpt" onclick='openSourceModal(${JSON.stringify(s).replace(/'/g, "&apos;")})'>🎬 ${esc(label)} <span class="src-score">${s.score}</span></div>`;
  }).join("");

  card.innerHTML = `
    <div class="notes-card-header">
      <div class="notes-topic-pill">📝 <span>${esc(data.topic)}</span></div>
      <div class="notes-card-actions">
        <button class="copy-btn" onclick="copyNotes(this)" title="Copy notes to clipboard">📋 Copy</button>
        <button class="download-btn" onclick="downloadNotes(this)" title="Download notes as markdown">📥 Download</button>
      </div>
    </div>
    <div class="notes-card-body">
      <div class="md">${marked.parse(data.notes || "")}</div>
    </div>
    ${sourcesHtml ? `
    <div class="notes-sources">
      <div class="notes-src-label">Lectures Cited in Notes</div>
      <div class="source-row">${sourcesHtml}</div>
    </div>` : ""}
  `;

  container.prepend(card);
  container.scrollTop = 0;
}

function copyNotes(btn) {
  const card = btn.closest(".notes-card");
  const text = card?.dataset.rawText || "";
  if (!text) return;

  navigator.clipboard.writeText(text)
    .then(() => {
      btn.textContent = "✅ Copied!";
      showToast("Notes copied to clipboard", "success");
      setTimeout(() => { btn.textContent = "📋 Copy"; }, 2000);
    })
    .catch(() => showToast("Copy failed", "error"));
}

function downloadNotes(btn) {
  const card = btn.closest(".notes-card");
  const text = card?.dataset.rawText || "";
  const topic = card?.dataset.topic || "study_notes";
  if (!text) return;

  const filename = topic.toLowerCase().replace(/[^a-z0-9]+/g, "_") + "_notes.md";
  const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast(`Downloaded ${filename}`, "success");
}

function copyText(btn, content) {
  navigator.clipboard.writeText(content)
    .then(() => {
      btn.textContent = "✅ Copied";
      showToast("Copied to clipboard", "success");
      setTimeout(() => { btn.textContent = "📋 Copy"; }, 2000);
    })
    .catch(() => showToast("Copy failed", "error"));
}

// ============================================================
// SOURCE EXCERPT MODAL
// ============================================================
function openSourceModal(sourceObj) {
  activeModalSource = sourceObj;
  const modal = document.getElementById("source-modal");
  const title = document.getElementById("modal-source-title");
  const chunkId = document.getElementById("modal-chunk-id");
  const score = document.getElementById("modal-score");
  const text = document.getElementById("modal-source-text");

  if (!modal) return;

  if (title) title.textContent = cleanName(sourceObj.source);
  if (chunkId) chunkId.textContent = `Chunk #${sourceObj.chunk_id || 0}`;
  if (score) score.textContent = `Relevance Score: ${sourceObj.score}`;
  if (text) text.textContent = sourceObj.full_text || sourceObj.preview || "No preview available.";

  modal.classList.add("active");
}

function hideSourceModal() {
  const modal = document.getElementById("source-modal");
  if (modal) modal.classList.remove("active");
  activeModalSource = null;
}

function closeSourceModal(e) {
  if (e.target.id === "source-modal") {
    hideSourceModal();
  }
}

function copySourceModalText() {
  if (!activeModalSource) return;
  const content = activeModalSource.full_text || activeModalSource.preview || "";
  navigator.clipboard.writeText(content)
    .then(() => showToast("Transcript excerpt copied", "success"))
    .catch(() => showToast("Copy failed", "error"));
}

// ============================================================
// TOAST NOTIFICATIONS
// ============================================================
function showToast(message, type = "info") {
  const container = document.getElementById("toasts");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  const icons = { success: "✅", error: "❌", info: "ℹ️" };
  toast.innerHTML = `<span>${icons[type] || "ℹ️"}</span> <span>${esc(message)}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = "toast-out 0.3s var(--ease) forwards";
    setTimeout(() => toast.remove(), 320);
  }, 3500);
}

// ============================================================
// UTILITY HELPERS
// ============================================================
function esc(str) {
  const d = document.createElement("div");
  d.appendChild(document.createTextNode(String(str || "")));
  return d.innerHTML;
}

function cleanName(src) {
  if (!src) return "Lecture Transcript";
  return src
    .replace(/\.txt$/i, "")
    .replace(/^\d+\.\s*/, "")
    .trim();
}
