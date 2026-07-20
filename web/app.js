(() => {
  const form = document.getElementById("ask-form");
  const input = document.getElementById("question-input");
  const submitBtn = form.querySelector(".inputbar__submit");
  const submitLabel = submitBtn.querySelector("span");
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
    const toggle = container.querySelector(".turn__sources-toggle");
    const list = container.querySelector(".turn__sources-list");
    list.innerHTML = "";
    list.hidden = true;
    toggle.setAttribute("aria-expanded", "false");
    if (!sources.length) {
      container.hidden = true;
      return;
    }
    container.hidden = false;
    sources.forEach((s, i) => {
      const row = sourceRowTpl.content.cloneNode(true);
      const srow = row.querySelector(".srow");
      srow.dataset.index = String(i + 1);
      const nameEl = row.querySelector(".srow__name");
      const displayName = s.label || s.source;
      nameEl.textContent = displayName;
      nameEl.title = displayName === s.source ? s.source : `${displayName} · ${s.source}`;
      list.appendChild(row);
    });
  }

  // Strips [n] citation markers from the text shown to the user — the
  // Sources list below is the reference, not inline brackets in the prose.
  function stripCitations(text) {
    return text
      .replace(/\[\d+\]/g, "")
      .replace(/ {2,}/g, " ")
      .replace(/ +([.,;:!?])/g, "$1")
      .trim();
  }

  function citedIndices(text) {
    return new Set([...text.matchAll(/\[(\d+)\]/g)].map((m) => m[1]));
  }

  function setAnswer(botText, text) {
    botText.textContent = stripCitations(text);
  }

  // Hides any source row the model didn't actually cite, so the list matches
  // what's really behind the answer instead of every chunk that was retrieved.
  function filterSourcesToCited(sourcesBox, text) {
    const cited = citedIndices(text);
    const rows = sourcesBox.querySelectorAll(".srow");
    let anyVisible = false;
    rows.forEach((row) => {
      const show = cited.has(row.dataset.index);
      row.hidden = !show;
      if (show) anyVisible = true;
    });
    sourcesBox.hidden = !anyVisible;
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
    setAnswer(botText, t.a);
    filterSourcesToCited(sourcesBox, t.a);
    return article;
  }

  function restoreTranscript() {
    if (!state.turns.length) return;
    hero.style.display = "none";
    for (const t of state.turns) renderCompletedTurn(t);
    const last = state.turns[state.turns.length - 1];
    setStatus(last.grounded ? "grounded" : "nomatch", last.grounded ? "ANSWERED" : "NO MATCH");
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
    setAnswer(botText, data.answer);
    filterSourcesToCited(sourcesBox, data.answer);
    setStatus(data.grounded ? "grounded" : "nomatch", data.grounded ? "ANSWERED" : "NO MATCH");
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
        setStatus(grounded ? "grounded" : "nomatch", grounded ? "ANSWERED" : "NO MATCH");
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
        setAnswer(botText, text);
        filterSourcesToCited(sourcesBox, text);
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
    submitLabel.textContent = "…";

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
      submitLabel.textContent = "ASK";
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

    const sourcesToggle = ev.target.closest(".turn__sources-toggle");
    if (sourcesToggle) {
      const list = sourcesToggle.parentElement.querySelector(".turn__sources-list");
      const expanded = sourcesToggle.getAttribute("aria-expanded") === "true";
      sourcesToggle.setAttribute("aria-expanded", String(!expanded));
      list.hidden = expanded;
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


