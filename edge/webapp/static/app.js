// NeuroEdge UI — no external dependencies; CSP allows only this file.
(function () {
  "use strict";
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";

  // ---------------------------------------------------------------- upload page
  const dz = document.getElementById("dropzone");
  const input = document.getElementById("scan");
  const fileLabel = document.getElementById("dz-file");
  const form = document.getElementById("upload-form");
  if (dz && input) {
    const show = () => { if (input.files.length) fileLabel.textContent = input.files[0].name; };
    input.addEventListener("change", show);
    ["dragenter", "dragover"].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.remove("drag"); }));
    dz.addEventListener("drop", ev => { if (ev.dataTransfer.files.length) { input.files = ev.dataTransfer.files; show(); } });
    form.addEventListener("submit", () => { const b = document.getElementById("submit"); b.disabled = true; b.textContent = "Analyzing…"; });
  }

  const postJSON = async (url, body) => {
    const r = await fetch(url, { method: "POST", headers: { "X-CSRF-Token": csrf, "Content-Type": "application/json" },
                                 body: JSON.stringify(body || {}) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || `Request failed (${r.status})`);
    return j;
  };
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

  // ---------------------------------------------------------------- report page
  const report = document.getElementById("report");
  if (report) {
    document.getElementById("print-btn")?.addEventListener("click", () => window.print());
    const nb = document.getElementById("narrative-btn");
    nb?.addEventListener("click", async () => {
      const out = document.getElementById("narrative");
      const label = nb.textContent;
      nb.disabled = true; nb.textContent = "Generating (local LLM)…";
      try {
        const j = await postJSON(`/case/${report.dataset.cid}/narrative`);
        out.textContent = j.narrative || j.message;
        out.classList.toggle("muted", !j.narrative);
      } catch (e) { out.textContent = e.message; }
      finally { nb.disabled = false; nb.textContent = label.trim(); }
    });
  }

  // ---------------------------------------------------------------- ask box
  const askForm = document.getElementById("ask-form");
  if (askForm) {
    const cid = document.getElementById("ask").dataset.cid;
    const q = document.getElementById("ask-q"), btn = document.getElementById("ask-btn"), log = document.getElementById("ask-log");
    askForm.addEventListener("submit", async ev => {
      ev.preventDefault();
      const question = q.value.trim(); if (!question) return;
      const item = el("div", "ask-item");
      item.append(el("p", "ask-q", question));
      const ans = el("div", "ask-a muted", "Thinking (local LLM)…");
      item.append(ans); log.prepend(item);
      btn.disabled = true; q.value = "";
      try {
        const j = await postJSON(`/case/${cid}/ask`, { question });
        ans.textContent = ""; ans.classList.remove("muted");
        if (j.answer) {
          ans.append(el("p", null, j.answer));
          ans.append(el("p", "badge ok", j.route ? "standard answer generated from the model output (no LLM)"
                                                 : "✓ checked against model output and sources"));
        } else {
          ans.append(el("p", "muted", j.llm_used ? "The assistant's answer failed the grounding check, so it is not shown. Relevant sources:"
                                                 : "Local LLM is offline. Relevant sources:"));
        }
        if (j.sources?.length) {
          const d = el("details", "small"); d.append(el("summary", null, `Sources (${j.sources.length})`));
          j.sources.forEach(s => { const p = el("p"); p.append(el("b", null, `[${s.source}] `), s.text); d.append(p); });
          if (!j.answer) d.open = true;
          ans.append(d);
        }
      } catch (e) { ans.textContent = e.message; }
      finally { btn.disabled = false; q.focus(); }
    });
  }

  // ---------------------------------------------------------------- result page
  const viewer = document.getElementById("viewer");
  if (!viewer) return;
  const overlay = document.getElementById("overlay");
  const opacity = document.getElementById("opacity");
  const gammaCtl = document.getElementById("gamma-ctl");
  const gamma = document.getElementById("gamma");
  const gammaVal = document.getElementById("gamma-val");
  const spinner = document.getElementById("spinner");
  const legends = { cam: "legend", finer: "legend-finer", lesion: "legend-lesion", deep: "legend-deep" };
  let deepUrl = null, timer = null;

  const finerUrl = () => overlay.dataset.finer + "?gamma=" + (gamma.value / 100).toFixed(2);
  const setLegend = kind => Object.entries(legends).forEach(([k, id]) => {
    const el = document.getElementById(id); if (el) el.classList.toggle("hidden", k !== kind);
  });

  async function runDeep() {
    spinner.classList.remove("hidden");
    try {
      const r = await fetch(`/case/${viewer.dataset.cid}/deep`, { method: "POST", headers: { "X-CSRF-Token": csrf } });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "Deep Explain failed");
      deepUrl = j.url;
      overlay.src = deepUrl;
    } catch (e) { alert(e.message); }
    finally { spinner.classList.add("hidden"); }
  }

  document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t === tab));
    const kind = tab.dataset.kind;
    gammaCtl.classList.toggle("hidden", kind !== "finer");
    setLegend(kind);
    if (kind === "finer") overlay.src = finerUrl();
    else if (kind === "deep") { if (deepUrl) overlay.src = deepUrl; else runDeep(); }
    else overlay.src = overlay.dataset[kind];
  }));

  opacity.addEventListener("input", () => { overlay.style.opacity = opacity.value / 100; });
  gamma.addEventListener("input", () => {
    gammaVal.textContent = (gamma.value / 100).toFixed(2);
    clearTimeout(timer);
    timer = setTimeout(() => { overlay.src = finerUrl(); }, 150);
  });
})();
