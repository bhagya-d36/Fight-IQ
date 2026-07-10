(() => {
  const form = document.getElementById("ask-form");
  const input = document.getElementById("question-input");
  const submitBtn = form.querySelector(".inputbar__submit");
  const transcript = document.getElementById("transcript");
  const hero = document.getElementById("hero");
  const chipRow = document.getElementById("suggested-chips");
  const statusChip = document.getElementById("status-chip");
  const statusText = document.getElementById("status-text");
  const newChatBtn = document.getElementById("new-chat");
  const turnTpl = document.getElementById("tpl-turn");
  const sourceRowTpl = document.getElementById("tpl-source-row");

  const STORAGE_KEY = "fightiq.chat.v1";
  const MAX_STORED_TURNS = 50;

  function freshState() {
    return { sessionId: crypto.randomUUID(), turns: [] };
  }

  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed.sessionId !== "string" || !Array.isArray(parsed.turns)) {
        return null;
      }
      return parsed;
    } catch {
      return null;
    }
  }

  function saveState() {
    try {
      const trimmed = { ...state, turns: state.turns.slice(-MAX_STORED_TURNS) };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
    } catch {
      // localStorage unavailable (quota, private mode) — degrade to no persistence.
    }
  }

  let state = loadState() || freshState();

  function setStatus(statusState, label) {
    statusChip.dataset.state = statusState;
    statusText.textContent = label;
  }

  function renderSources(container, sources) {
    const label = container.querySelector(".turn__sources-label");
    const list = container.querySelector(".turn__sources-list");
    list.innerHTML = "";
    if (!sources.length) {
      container.hidden = true;
      return;
    }
    container.hidden = false;
    label.textContent = "SOURCES";
    sources.forEach((s, i) => {
      const row = sourceRowTpl.content.cloneNode(true);
      const srow = row.querySelector(".srow");
      srow.dataset.index = String(i + 1);
      row.querySelector(".srow__name").textContent = s.source;
      row.querySelector(".srow__score").textContent = s.score.toFixed(2);
      row.querySelector(".srow__bar-fill").style.width = `${Math.round(s.score * 100)}%`;
      list.appendChild(row);
    });
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  // Renders the final answer text, linkifying [n] citation markers that fall
  // within range of the sources list (out-of-range markers stay plain text).
  function setAnswer(botText, text, sources) {
    const n = sources.length;
    botText.innerHTML = escapeHtml(text).replace(/\[(\d+)\]/g, (match, digits) => {
      const i = parseInt(digits, 10);
      return i >= 1 && i <= n ? `<button type="button" class="cite" data-index="${i}">[${i}]</button>` : match;
    });
  }

  function newTurn(question) {
    const node = turnTpl.content.cloneNode(true);
    node.querySelector(".turn__you-text").textContent = question;
    const article = node.querySelector(".turn");
    transcript.appendChild(node);
    return article;
  }

  function renderCompletedTurn(t) {
    const article = newTurn(t.q);
    const botText = article.querySelector(".turn__bot-text");
    const sourcesBox = article.querySelector(".turn__sources");
    const sources = t.sources || [];
    renderSources(sourcesBox, sources);
    setAnswer(botText, t.a, sources);
    return article;
  }

  function restoreTranscript() {
    if (!state.turns.length) return;
    hero.style.display = "none";
    for (const t of state.turns) renderCompletedTurn(t);
    const last = state.turns[state.turns.length - 1];
    setStatus(last.grounded ? "grounded" : "nomatch", last.grounded ? "GROUNDED" : "NO MATCH");
    transcript.parentElement.scrollTop = transcript.parentElement.scrollHeight;
  }

  async function askViaJson(question, botText, sourcesBox) {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: state.sessionId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.answer || "request failed");
    const sources = data.sources || [];
    renderSources(sourcesBox, sources);
    setAnswer(botText, data.answer, sources);
    setStatus(data.grounded ? "grounded" : "nomatch", data.grounded ? "GROUNDED" : "NO MATCH");
    return { answer: data.answer, grounded: data.grounded, sources };
  }

  function askViaStream(question, botText, sourcesBox) {
    return new Promise((resolve, reject) => {
      const url = `/api/ask/stream?q=${encodeURIComponent(question)}&session_id=${encodeURIComponent(state.sessionId)}`;
      const es = new EventSource(url);
      let text = "";
      let sources = [];
      let grounded = false;
      let gotSources = false;
      let settled = false;

      es.addEventListener("sources", (ev) => {
        const data = JSON.parse(ev.data);
        gotSources = true;
        grounded = !!data.grounded;
        sources = data.sources || [];
        renderSources(sourcesBox, sources);
        setStatus(grounded ? "grounded" : "nomatch", grounded ? "GROUNDED" : "NO MATCH");
      });

      es.addEventListener("token", (ev) => {
        const data = JSON.parse(ev.data);
        text += data.text;
        botText.textContent = text;
        botText.classList.remove("is-empty");
        transcript.parentElement.scrollTop = transcript.parentElement.scrollHeight;
      });

      es.addEventListener("done", () => {
        settled = true;
        es.close();
        if (!gotSources) {
          // Server hit an error before emitting "sources" — surface it as a failure.
          reject(new Error(text || "stream failed"));
          return;
        }
        setAnswer(botText, text, sources);
        resolve({ answer: text, grounded, sources });
      });

      es.onerror = () => {
        es.close();
        if (!settled) reject(new Error("stream failed"));
      };
    });
  }

  async function runTurn(question, article) {
    const botText = article.querySelector(".turn__bot-text");
    const sourcesBox = article.querySelector(".turn__sources");
    const errorBox = article.querySelector(".turn__error");
    const errorText = errorBox.querySelector(".turn__error-text");

    errorBox.hidden = true;
    botText.textContent = "";
    botText.classList.add("is-empty");
    setStatus("thinking", "THINKING");
    input.disabled = true;
    submitBtn.disabled = true;

    let result = null;
    try {
      if (typeof EventSource !== "undefined") {
        result = await askViaStream(question, botText, sourcesBox);
      } else {
        result = await askViaJson(question, botText, sourcesBox);
      }
    } catch {
      try {
        result = await askViaJson(question, botText, sourcesBox);
      } catch {
        errorText.textContent = "Something went wrong reaching the assistant.";
        errorBox.hidden = false;
        setStatus("nomatch", "ERROR");
      }
    } finally {
      botText.classList.remove("is-empty");
      input.disabled = false;
      submitBtn.disabled = false;
      input.focus();
      transcript.parentElement.scrollTop = transcript.parentElement.scrollHeight;
    }

    if (result) {
      state.turns.push({ q: question, a: result.answer, grounded: result.grounded, sources: result.sources });
      saveState();
    }
  }

  function handleAsk(question) {
    hero.style.display = "none";
    const article = newTurn(question);
    input.value = "";
    runTurn(question, article);
  }

  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    handleAsk(question);
  });

  chipRow.addEventListener("click", (ev) => {
    const chip = ev.target.closest(".chip");
    if (!chip) return;
    handleAsk(chip.dataset.q);
  });

  transcript.addEventListener("click", (ev) => {
    const retryBtn = ev.target.closest(".turn__retry");
    if (retryBtn) {
      const article = retryBtn.closest(".turn");
      const question = article.querySelector(".turn__you-text").textContent;
      runTurn(question, article);
      return;
    }

    const cite = ev.target.closest(".cite");
    if (cite) {
      const article = cite.closest(".turn");
      const row = article.querySelector(`.srow[data-index="${cite.dataset.index}"]`);
      if (row) {
        row.classList.add("srow--flash");
        row.scrollIntoView({ block: "nearest", behavior: "smooth" });
        setTimeout(() => row.classList.remove("srow--flash"), 1200);
      }
    }
  });

  newChatBtn.addEventListener("click", () => {
    state = freshState();
    saveState();
    for (const article of transcript.querySelectorAll(".turn")) article.remove();
    hero.style.display = "";
    setStatus("idle", "READY");
    input.value = "";
    input.focus();
  });

  restoreTranscript();
})();
