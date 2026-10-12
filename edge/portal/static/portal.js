// NeuroEdge portal — no external dependencies; the CSP allows only this file.
(function () {
  "use strict";
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const postJSON = async (url, body) => {
    const r = await fetch(url, { method: "POST", headers: { "X-CSRF-Token": csrf, "Content-Type": "application/json" },
                                 body: JSON.stringify(body || {}) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || `Request failed (${r.status})`);
    return j;
  };

  // ------------------------------------------------------------------ upload (drag & drop)
  const dz = document.getElementById("dropzone"), input = document.getElementById("scan");
  if (dz && input) {
    const label = document.getElementById("dz-file");
    const show = () => { if (input.files.length) label.textContent = input.files[0].name; };
    input.addEventListener("change", show);
    ["dragenter", "dragover"].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.remove("drag"); }));
    dz.addEventListener("drop", ev => { if (ev.dataTransfer.files.length) { input.files = ev.dataTransfer.files; show(); } });
    document.getElementById("upload-form")?.addEventListener("submit", () => {
      const b = document.getElementById("submit"); b.disabled = true; b.textContent = "Uploading and analysing…";
    });
  }

  // ------------------------------------------------------------------ doctor review: viewer
  const review = document.getElementById("review");
  if (!review) return;
  const cid = review.dataset.cid;
  const overlay = document.getElementById("overlay");
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach(tab => tab.addEventListener("click", () => {
    tabs.forEach(t => t.classList.toggle("active", t === tab));
    overlay.src = overlay.dataset.base.replace(/X\.png$/, tab.dataset.kind + ".png");
  }));
  document.getElementById("opacity")?.addEventListener("input", e => { overlay.style.opacity = e.target.value / 100; });

  // ------------------------------------------------------------------ assistant
  const nb = document.getElementById("narrative-btn"), out = document.getElementById("narrative");
  nb?.addEventListener("click", async () => {
    nb.disabled = true; const label = nb.textContent; nb.textContent = "Drafting (local LLM)…";
    out.textContent = "";
    try {
      const j = await postJSON(`/doctor/case/${cid}/narrative`);
      if (j.narrative) out.append(el("p", null, j.narrative));
      else {
        out.append(el("p", "muted small", j.message));
        (j.sections || []).forEach(([t, txt]) => { const p = el("p"); p.append(el("b", null, t + ": "), txt); out.append(p); });
      }
    } catch (e) { out.append(el("p", "muted", e.message)); }
    finally { nb.disabled = false; nb.textContent = label; }
  });
  const form = document.getElementById("ask-form"), q = document.getElementById("ask-q"), log = document.getElementById("ask-log");
  form?.addEventListener("submit", async ev => {
    ev.preventDefault();
    const question = q.value.trim(); if (!question) return;
    const item = el("div", "qa"); item.append(el("p", "q", question));
    const ans = el("div", "muted small", "Thinking…"); item.append(ans); log.prepend(item); q.value = "";
    try {
      const j = await postJSON(`/doctor/case/${cid}/ask`, { question });
      ans.className = ""; ans.textContent = "";
      if (j.answer) {
        ans.append(el("p", null, j.answer));
        ans.append(el("p", "muted small", j.route ? "Standard answer from the model output (no LLM)." : "Checked against the model output and sources."));
      } else ans.append(el("p", "muted small", j.llm_used ? "The draft answer failed the grounding check. Sources:" : "Local LLM offline. Relevant sources:"));
      if (j.sources?.length) {
        const d = el("details", "small"); d.append(el("summary", null, `Sources (${j.sources.length})`));
        j.sources.forEach(s => { const p = el("p"); p.append(el("b", null, `[${s.source}] `), s.text); d.append(p); });
        ans.append(d);
      }
    } catch (e) { ans.textContent = e.message; }
  });
})();
