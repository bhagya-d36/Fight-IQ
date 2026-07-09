(() => {
  const form = document.getElementById("ask-form");
  const input = document.getElementById("question-input");
  const transcript = document.getElementById("transcript");
  const hero = document.getElementById("hero");
  const chipRow = document.getElementById("suggested-chips");
  const statusChip = document.getElementById("status-chip");
  const statusText = document.getElementById("status-text");
  const turnTpl = document.getElementById("tpl-turn");
  const sourceRowTpl = document.getElementById("tpl-source-row");

  function setStatus(state, label) {
    statusChip.dataset.state = state;
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
    for (const s of sources) {
      const row = sourceRowTpl.content.cloneNode(true);
      row.querySelector(".srow__name").textContent = s.source;
      row.querySelector(".srow__score").textContent = s.score.toFixed(2);
      row.querySelector(".srow__bar-fill").style.width = `${Math.round(s.score * 100)}%`;
      list.appendChild(row);
    }
  }

  function newTurn(question) {
    const node = turnTpl.content.cloneNode(true);
    node.querySelector(".turn__you-text").textContent = question;
    const article = node.querySelector(".turn");
    transcript.appendChild(node);
    return article;
  }

  async function askViaJson(question, botText, sourcesBox) {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    botText.textContent = data.answer;
    renderSources(sourcesBox, data.sources || []);
    setStatus(data.grounded ? "grounded" : "nomatch", data.grounded ? "GROUNDED" : "NO MATCH");
  }

  function askViaStream(question, botText, sourcesBox) {
    return new Promise((resolve, reject) => {
      const es = new EventSource(`/api/ask/stream?q=${encodeURIComponent(question)}`);
      let text = "";
      let settled = false;

      es.addEventListener("sources", (ev) => {
        const data = JSON.parse(ev.data);
        renderSources(sourcesBox, data.sources || []);
        setStatus(data.grounded ? "grounded" : "nomatch", data.grounded ? "GROUNDED" : "NO MATCH");
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
        resolve();
      });

      es.onerror = () => {
        es.close();
        if (!settled) reject(new Error("stream failed"));
      };
    });
  }

  async function handleAsk(question) {
    hero.style.display = "none";
    setStatus("thinking", "THINKING");

    const article = newTurn(question);
    const botText = article.querySelector(".turn__bot-text");
    const sourcesBox = article.querySelector(".turn__sources");
    botText.classList.add("is-empty");

    input.value = "";
    input.disabled = true;

    try {
      if (typeof EventSource !== "undefined") {
        await askViaStream(question, botText, sourcesBox);
      } else {
        await askViaJson(question, botText, sourcesBox);
      }
    } catch (err) {
      try {
        await askViaJson(question, botText, sourcesBox);
      } catch (err2) {
        botText.textContent = "Something went wrong reaching the assistant.";
        setStatus("nomatch", "ERROR");
      }
    } finally {
      botText.classList.remove("is-empty");
      input.disabled = false;
      input.focus();
      transcript.parentElement.scrollTop = transcript.parentElement.scrollHeight;
    }
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
})();
