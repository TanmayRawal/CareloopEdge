// NeuroEdge Imaging portal — no external dependencies; the CSP allows only this file.
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const csrf = $('meta[name="csrf-token"]')?.content || "";
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const postJSON = async (url, body) => {
    const r = await fetch(url, { method: "POST", headers: { "X-CSRF-Token": csrf, "Content-Type": "application/json" },
                                 body: JSON.stringify(body || {}) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || `Request failed (${r.status})`);
    return j;
  };

  // ------------------------------------------------------------------ login: patient / clinician tabs
  const tabs = $$(".role-tabs button");
  tabs.forEach(b => b.addEventListener("click", () => {
    tabs.forEach(t => { t.classList.toggle("on", t === b); t.setAttribute("aria-selected", t === b); });
    const staff = b.dataset.role === "staff";
    $("#alt-patient")?.classList.toggle("hide", staff);
    $("#alt-staff")?.classList.toggle("hide", !staff);
    const lead = $("#login-lead");
    if (lead) lead.textContent = staff ? "Radiologists and administrators." : "Access your reports and scans.";
  }));

  // ------------------------------------------------------------------ clickable table rows
  $$("tr.link[data-href]").forEach(tr => tr.addEventListener("click", ev => {
    if (ev.target.closest("a, button, form, input")) return;
    window.location = tr.dataset.href;
  }));

  // ------------------------------------------------------------------ print
  $("#print-btn")?.addEventListener("click", () => window.print());

  // ------------------------------------------------------------------ upload (drag & drop + steps)
  const dz = $("#dropzone"), input = $("#scan");
  if (dz && input) {
    const label = $("#dz-file");
    const show = () => {
      if (!input.files.length) return;
      label.textContent = "✓ " + input.files[0].name;
      $("#st1")?.classList.remove("on"); $("#st2")?.classList.add("on");
    };
    input.addEventListener("change", show);
    ["dragenter", "dragover"].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.remove("drag"); }));
    dz.addEventListener("drop", ev => { if (ev.dataTransfer.files.length) { input.files = ev.dataTransfer.files; show(); } });
    $("#upload-form")?.addEventListener("change", ev => {
      if (ev.target.name === "consent" && ev.target.checked) { $("#st2")?.classList.remove("on"); $("#st3")?.classList.add("on"); }
    });
    $("#upload-form")?.addEventListener("submit", () => {
      const b = $("#submit"); b.disabled = true; b.textContent = "Uploading securely…";
    });
  }

  // ------------------------------------------------------------------ case review: viewer
  const review = $("#review");
  if (!review) return;
  const cid = review.dataset.cid;
  const stage = $("#stage"), canvas = $("#canvas"), overlay = $("#overlay"), zoomLabel = $("#zoom-label");
  let scale = 1, tx = 0, ty = 0;
  const apply = () => {
    canvas.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    if (zoomLabel) zoomLabel.textContent = Math.round(scale * 100) + "%";
  };
  const zoomAt = (factor, cx, cy) => {
    const r = stage.getBoundingClientRect();
    cx = cx ?? r.width / 2; cy = cy ?? r.height / 2;
    const ns = Math.min(8, Math.max(1, scale * factor));
    tx = cx - (cx - tx) * (ns / scale); ty = cy - (cy - ty) * (ns / scale); scale = ns;
    if (scale === 1) { tx = 0; ty = 0; }
    apply();
  };
  const reset = () => { scale = 1; tx = 0; ty = 0; apply(); };
  stage.addEventListener("wheel", ev => {
    ev.preventDefault();
    const r = stage.getBoundingClientRect();
    zoomAt(ev.deltaY < 0 ? 1.15 : 1 / 1.15, ev.clientX - r.left, ev.clientY - r.top);
  }, { passive: false });
  let drag = null;
  stage.addEventListener("pointerdown", ev => { if (scale > 1) { drag = { x: ev.clientX - tx, y: ev.clientY - ty }; stage.setPointerCapture(ev.pointerId); stage.classList.add("grabbing"); } });
  stage.addEventListener("pointermove", ev => { if (drag) { tx = ev.clientX - drag.x; ty = ev.clientY - drag.y; apply(); } });
  ["pointerup", "pointercancel"].forEach(e => stage.addEventListener(e, () => { drag = null; stage.classList.remove("grabbing"); }));
  stage.addEventListener("dblclick", reset);

  const modes = $$(".v-modes button");
  const setMode = btn => {
    modes.forEach(m => m.classList.toggle("on", m === btn));
    overlay.src = overlay.dataset.base.replace(/X\.png$/, btn.dataset.kind + ".png");
  };
  modes.forEach(b => b.addEventListener("click", () => setMode(b)));
  const toggleExpand = () => { review.classList.toggle("expanded"); };
  $$(".v-tool").forEach(b => b.addEventListener("click", () => {
    ({ in: () => zoomAt(1.25), out: () => zoomAt(0.8), reset, expand: toggleExpand })[b.dataset.act]?.();
  }));
  $("#opacity")?.addEventListener("input", e => { overlay.style.opacity = e.target.value / 100; });
  document.addEventListener("keydown", ev => {
    if ((ev.target instanceof Element && ev.target.closest("input, textarea, select, [contenteditable]")) || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (ev.key >= "1" && ev.key <= "4" && modes[+ev.key - 1]) setMode(modes[+ev.key - 1]);
    else if (ev.key === "+" || ev.key === "=") zoomAt(1.25);
    else if (ev.key === "-") zoomAt(0.8);
    else if (ev.key === "0") reset();
    else if (ev.key.toLowerCase() === "f") toggleExpand();
    else return;
    ev.preventDefault();
  });

  // ------------------------------------------------------------------ assistant
  const nb = $("#narrative-btn"), out = $("#narrative");
  nb?.addEventListener("click", async () => {
    nb.disabled = true; const label = nb.textContent; nb.textContent = "Drafting…";
    out.textContent = "";
    try {
      const j = await postJSON(`/doctor/case/${cid}/narrative`);
      if (j.narrative) out.append(el("p", null, j.narrative));
      else {
        if (j.message) out.append(el("p", "muted small", j.message));
        (j.sections || []).forEach(([t, txt]) => { const p = el("p"); p.append(el("b", null, t + ": "), txt); out.append(p); });
      }
    } catch (e) { out.append(el("p", "muted", e.message)); }
    finally { nb.disabled = false; nb.textContent = label; }
  });
  const form = $("#ask-form"), q = $("#ask-q"), log = $("#ask-log");
  form?.addEventListener("submit", async ev => {
    ev.preventDefault();
    const question = q.value.trim(); if (!question) return;
    const item = el("div", "qa"); item.append(el("p", "q", question));
    const ans = el("div", "muted small", "Thinking…"); item.append(ans); log.prepend(item); q.value = "";
    try {
      const j = await postJSON(`/doctor/case/${cid}/ask`, { question });
      ans.className = ""; ans.textContent = "";
      if (j.answer) ans.append(el("p", null, j.answer));
      else ans.append(el("p", "muted small", j.llm_used ? "No verified answer; related sources:" : "Related sources:"));
      if (j.sources?.length) {
        const d = el("details", "small"); d.append(el("summary", null, `Sources (${j.sources.length})`));
        j.sources.forEach(s => { const p = el("p"); p.append(el("b", null, `[${s.source}] `), s.text); d.append(p); });
        ans.append(d);
      }
    } catch (e) { ans.textContent = e.message; }
  });
})();
