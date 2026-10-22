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
  if (new URLSearchParams(location.search).get("as") === "staff") tabs.find(t => t.dataset.role === "staff")?.click();

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
      label.textContent = "✓ " + (input.files.length > 1 ? `${input.files.length} files (one series)` : input.files[0].name);
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

  // ------------------------------------------------------------------ upload progress (a whole study takes 10-30 s)
  $$('form[enctype="multipart/form-data"]').forEach(f => f.addEventListener("submit", () => {
    const files = $$('input[type=file][name="scan"]', f).flatMap(i => Array.from(i.files));
    if (!files.length) return;
    const series = files.length > 1 || files.some(x => /\.(zip|nii|nii\.gz)$/i.test(x.name));
    $("#busy-t").textContent = series ? "Uploading and analysing the study…" : "Uploading and analysing the image…";
    if (!series) $("#busy-d").textContent = "This takes a few seconds. Please keep this page open.";
    $("#busy")?.classList.add("on");
  }));

  // ------------------------------------------------------------------ device home + kiosk
  const clock = $("#dv-time");
  if (clock) {
    const tick = () => {
      const d = new Date();
      clock.textContent = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      $("#dv-date").textContent = d.toLocaleDateString([], { weekday: "long", day: "numeric", month: "long", year: "numeric" });
    };
    tick(); setInterval(tick, 10000);
    setTimeout(() => location.reload(), 60000);           // status (temperature, last analysis) stays current
  }
  if (document.body.classList.contains("kiosk")) {         // the device's own screen: back to device home when idle
    let idle;
    const arm = () => { clearTimeout(idle); idle = setTimeout(() => { location.href = "/device"; }, 3 * 60 * 1000); };
    ["pointerdown", "keydown", "wheel", "touchstart"].forEach(e => document.addEventListener(e, arm, { passive: true }));
    arm();
  }

  // ------------------------------------------------------------------ confirmations for destructive actions
  $$("form[data-confirm]").forEach(f => f.addEventListener("submit", ev => {
    if (!window.confirm(f.dataset.confirm)) ev.preventDefault();
  }));

  // ------------------------------------------------------------------ notification bell: close on outside click / Esc
  const bell = $("details.bell");
  if (bell) {
    document.addEventListener("click", ev => { if (bell.open && !bell.contains(ev.target)) bell.open = false; });
    document.addEventListener("keydown", ev => { if (ev.key === "Escape") bell.open = false; });
  }

  // ------------------------------------------------------------------ pop-over forms (log contact): one open at a time
  const pops = document.querySelectorAll("details.pop");
  if (pops.length) {
    pops.forEach(p => p.addEventListener("toggle", () => { if (p.open) pops.forEach(o => { if (o !== p) o.open = false; }); }));
    document.addEventListener("click", ev => pops.forEach(p => { if (p.open && !p.contains(ev.target)) p.open = false; }));
    document.addEventListener("keydown", ev => { if (ev.key === "Escape") pops.forEach(p => { p.open = false; }); });
  }

  // ------------------------------------------------------------------ sign panel: referral fields only for a referral
  const fuKind = $("#fu-kind"), fuRef = $("#fu-ref");
  if (fuKind && fuRef) {
    const sync = () => {
      const on = fuKind.value === "referral";
      fuRef.classList.toggle("on", on);
      const fac = fuRef.querySelector("input[name=ref_facility]");
      if (fac) fac.required = on;
    };
    fuKind.addEventListener("change", sync);
    sync();
  }

  // ------------------------------------------------------------------ booking: live summary of the chosen slot
  const bookForm = $("#book-form");
  if (bookForm) {
    const btn = $("#book-btn"), lbl = $("#sel-label"), room = $("#sel-room");
    bookForm.addEventListener("change", ev => {
      if (ev.target.name !== "start_ts") return;
      lbl.textContent = ev.target.dataset.label;
      room.textContent = ev.target.dataset.room;
      btn.disabled = false;
    });
    bookForm.addEventListener("submit", () => { btn.disabled = true; btn.textContent = "Booking…"; });
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
  const imgUrl = kind => overlay.dataset.base.replace(/X\.png$/, kind + ".png");
  // whole series: one slice at a time; Grad-CAM / lesion overlays exist for the key slices only
  const series = $("#series"), base = $("#base"), slider = $("#slice");
  const sKeys = series ? JSON.parse(series.dataset.keys) : [], sInfo = series ? JSON.parse(series.dataset.slices) : [];
  let slice = series ? +series.dataset.start : null, mode = modes.find(m => m.classList.contains("on")) || modes[0];
  const render = () => {
    if (slice === null) { overlay.src = imgUrl(mode.dataset.kind); return; }
    const att = series.dataset.att, isKey = sKeys.includes(slice), kind = mode.dataset.kind;
    base.src = imgUrl(`s${slice}_original_${att}`);
    overlay.src = imgUrl(isKey || series.dataset.all === "1" ? `s${slice}_${kind}` : `s${slice}_original_${att}`);
    slider.value = slice;
    const s = sInfo[slice];
    $("#slice-label").textContent = `${slice + 1}/${sInfo.length} · ${s.plane} ${s.index} · tumour ${Math.round(100 * s.tumour_prob)}%` +
      (isKey ? ` · key slice ${sKeys.indexOf(slice) + 1}` : "") + (s.ok ? "" : " · not usable");
    $$("#slice-bars span").forEach(b => b.classList.toggle("on", +b.dataset.i === slice));
    // each slice's Grad-CAM explains that slice's own prediction, not the study's answer
    const labels = JSON.parse(series.dataset.labels);
    const hasMap = isKey || series.dataset.all === "1";
    $("#cam-class").textContent = !hasMap ? "— (key slices only)"
      : (labels[s.prediction] || s.prediction) + (s.prediction !== sInfo[sKeys[0]].prediction ? " (this slice)" : "");
  };
  const setSlice = i => { slice = Math.max(0, Math.min(sInfo.length - 1, i)); render(); };
  slider?.addEventListener("input", () => setSlice(+slider.value));
  $$("#slice-bars span, .vs-keys button").forEach(b => b.addEventListener("click", () => setSlice(+(b.dataset.i ?? b.dataset.slice))));
  const setMode = btn => {
    mode = btn;
    modes.forEach(m => m.classList.toggle("on", m === btn));
    render();
  };
  $$("#slice-bars span").forEach(b => { b.style.height = b.dataset.h + "%"; });   // CSP: no inline style attributes
  if (series) render();

  // touch: two fingers pinch-zoom; one finger (not zoomed) swipes up/down through the slices of a series
  const touches = new Map();
  let pinch = null, swipeY = null;
  const SWIPE_PX = 24;
  stage.addEventListener("pointerdown", ev => {
    if (ev.pointerType === "mouse") return;
    touches.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    if (touches.size === 2) {
      const [a, b] = [...touches.values()];
      pinch = Math.hypot(a.x - b.x, a.y - b.y); drag = null; swipeY = null;
    } else if (scale === 1 && series) swipeY = ev.clientY;
  });
  stage.addEventListener("pointermove", ev => {
    if (!touches.has(ev.pointerId)) return;
    touches.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    if (touches.size === 2 && pinch) {
      const [a, b] = [...touches.values()], dist = Math.hypot(a.x - b.x, a.y - b.y), r = stage.getBoundingClientRect();
      zoomAt(dist / pinch, (a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top); pinch = dist;
    } else if (swipeY !== null && Math.abs(ev.clientY - swipeY) >= SWIPE_PX) {
      const steps = Math.trunc((ev.clientY - swipeY) / SWIPE_PX);
      setSlice(slice - steps); swipeY += steps * SWIPE_PX;      // finger up = next slice down the stack
    }
  });
  ["pointerup", "pointercancel"].forEach(e => stage.addEventListener(e, ev => {
    touches.delete(ev.pointerId);
    if (touches.size < 2) pinch = null;
    if (!touches.size) swipeY = null;
  }));
  modes.forEach(b => b.addEventListener("click", () => setMode(b)));
  const toggleExpand = () => { document.body.classList.toggle("case-expanded", review.classList.toggle("expanded")); };
  // compact screens: the case page is three tabs (Image | AI findings | Report); on large screens all show at once
  $$("#case-tabs button").forEach(b => b.addEventListener("click", () => {
    $$("#case-tabs button").forEach(t => { t.classList.toggle("on", t === b); t.setAttribute("aria-selected", t === b); });
    review.dataset.tab = b.dataset.tab;
  }));
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
    else if (series && (ev.key === "ArrowUp" || ev.key === "ArrowDown")) setSlice(slice + (ev.key === "ArrowUp" ? -1 : 1));
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
