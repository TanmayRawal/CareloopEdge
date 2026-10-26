// CareLoop-Edge portal — no external dependencies; the CSP allows only this file.
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

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

  // ------------------------------------------------------------------ report: structured findings templates
  $$("[data-tpl]").forEach(b => b.addEventListener("click", () => {
    const t = $("#findings");
    if (t.value.trim() && !confirm("Replace the findings with the template?")) return;
    t.value = b.dataset.tpl; t.focus();
  }));

  // ------------------------------------------------------------------ upload (drag & drop + steps)
  const dz = $("#dropzone"), input = $("#scan");
  if (dz && input) {
    const label = $("#dz-file");
    const show = () => {
      if (!input.files.length) return;
      label.textContent = (input.files.length > 1 ? `${input.files.length} files (one series)` : input.files[0].name);
      $("#st1")?.classList.remove("on"); $("#st2")?.classList.add("on");
    };
    input.addEventListener("change", show);
    const dir = $("#scan-dir");                      // a whole folder (DICOM series on a USB drive or CD)
    dir?.addEventListener("change", () => {
      if (!dir.files.length) return;
      input.value = ""; input.required = false;
      label.textContent = `${dir.files.length} files from the folder`;
      $("#st1")?.classList.remove("on"); $("#st2")?.classList.add("on");
    });
    input.addEventListener("change", () => { if (dir && input.files.length) { dir.value = ""; input.required = true; } });
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
    };
    tick(); setInterval(tick, 10000);
    setTimeout(() => location.reload(), 5 * 60000);       // status (temperature, storage) stays current
  }
  // ------------------------------------------------------------------ device home: 3D brain
  // Surface points of a real MRI brain (training/make_brain_points.py: int8 x, y, z, nx, ny, nz, RAS+), drawn on a
  // 2D canvas: turntable about the superior axis, camera 22 degrees above, back faces culled, Lambert shading in 8
  // tones, and an axial "scan" plane sweeping head to foot. No WebGL, no libraries; ~6k points per frame at <= 30 fps.
  const brain = $("#brain");
  if (brain) {
    const ctx = brain.getContext("2d");
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const TONES = 8, ELEV = 22 * Math.PI / 180, SPEED = 0.22;              // rad/s: one turn in ~29 s
    const L = [-0.35, 0.45, 0.82], ln = Math.hypot(...L); L.forEach((v, i) => { L[i] = v / ln; });
    const tone = k => {                                                     // accent hue, brightness by shading
      const t = 0.18 + 0.82 * k / (TONES - 1);
      return `rgb(${Math.round(30 + 60 * t)}, ${Math.round(80 + 130 * t)}, ${Math.round(100 + 130 * t)})`;
    };
    const palette = Array.from({ length: TONES }, (_, k) => tone(k));
    let P = null, n = 0, yaw = 0.6, pitch = ELEV, last = 0, drag = null, idleUntil = 0, W = 0, H = 0, dpr = 1;
    // stop animating after a minute without a touch (CPU and heat stay free for analysis); a touch resumes it
    const REST_MS = 60000;
    let touched = performance.now(), resting = false;
    const fit = () => {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      const r = brain.getBoundingClientRect();
      W = brain.width = Math.max(1, Math.round(r.width * dpr)); H = brain.height = Math.max(1, Math.round(r.height * dpr));
    };
    const draw = now => {
      if (!P) return;
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const scale = 0.47 * Math.min(W, H), ox = W / 2, oy = H / 2, s = Math.max(1.4, 1.9 * dpr * Math.min(W, H) / 520);
      const scan = still ? 2 : Math.sin(now / 2600) * 1.05;                 // axial plane position (z, -1..1)
      ctx.clearRect(0, 0, W, H);
      const xs = new Float32Array(n), ys = new Float32Array(n), tk = new Int8Array(n);
      for (let i = 0; i < n; i++) {
        const o = i * 6, x = P[o] / 127, y = P[o + 1] / 127, z = P[o + 2] / 127;
        const nx = P[o + 3] / 127, ny = P[o + 4] / 127, nz = P[o + 5] / 127;
        // yaw about superior (z), then pitch about screen x; camera looks along +y
        const x1 = x * cy - y * sy, y1 = x * sy + y * cy;
        const nx1 = nx * cy - ny * sy, ny1 = nx * sy + ny * cy;
        const z2 = y1 * sp + z * cp;
        const ny2 = ny1 * cp - nz * sp, nz2 = ny1 * sp + nz * cp;
        const toward = -ny2;                                              // normal component toward the viewer
        if (toward < -0.05) { tk[i] = -1; continue; }
        const lam = Math.max(0, nx1 * L[0] + nz2 * L[1] + toward * L[2]);
        xs[i] = ox + x1 * scale; ys[i] = oy - z2 * scale;
        tk[i] = Math.abs(z - scan) < 0.035 ? TONES : Math.min(TONES - 1, Math.floor(lam * TONES));
      }
      for (let k = 0; k <= TONES; k++) {
        ctx.fillStyle = k === TONES ? "#e9f8fc" : palette[k];
        const sz = k === TONES ? s * 1.25 : s;
        for (let i = 0; i < n; i++) if (tk[i] === k) ctx.fillRect(xs[i] - sz / 2, ys[i] - sz / 2, sz, sz);
      }
    };
    const loop = now => {
      if (now - touched > REST_MS && !drag) { resting = true; return; }
      if (document.hidden) { requestAnimationFrame(loop); return; }
      if (now - last >= 33) {                                             // ~30 fps is plenty, and cheap on a Pi
        if (!drag && now > idleUntil) yaw += SPEED * Math.min(0.1, (now - (last || now)) / 1000);
        last = now; draw(now);
      }
      requestAnimationFrame(loop);
    };
    const wake = () => { touched = performance.now(); if (resting && !still) { resting = false; last = 0; requestAnimationFrame(loop); } };
    ["pointerdown", "keydown", "touchstart"].forEach(e => document.addEventListener(e, wake, { passive: true }));
    brain.addEventListener("pointerdown", ev => { drag = { x: ev.clientX, y: ev.clientY, yaw, pitch }; brain.setPointerCapture(ev.pointerId); });
    brain.addEventListener("pointermove", ev => {
      if (!drag) return;
      yaw = drag.yaw + (ev.clientX - drag.x) * 0.01;
      pitch = Math.max(-0.6, Math.min(1.2, drag.pitch + (ev.clientY - drag.y) * 0.01));
      if (still) draw(performance.now());
    });
    ["pointerup", "pointercancel"].forEach(e => brain.addEventListener(e, () => { drag = null; idleUntil = performance.now() + 3000; }));
    new ResizeObserver(() => { fit(); draw(performance.now()); }).observe(brain);
    fetch(brain.dataset.src).then(r => r.arrayBuffer()).then(buf => {
      P = new Int8Array(buf); n = Math.floor(P.length / 6); fit();
      if (still) draw(0); else requestAnimationFrame(loop);
    }).catch(() => brain.remove());
  }

  // ------------------------------------------------------------------ kiosk: on-screen keyboard
  // The device's touch screen has no keyboard; the compositor's own keyboard is not reliable in a bare kiosk
  // session, so the portal brings one. Kiosk only (cookie), appears when a text field is focused.
  if (document.body.classList.contains("kiosk")) {
    const ROWS = {
      abc: ["1234567890", "qwertyuiop", "asdfghjkl", "⇧zxcvbnm⌫", "?123 @ . space _ - ↵"],
      sym: ["1234567890", "!#$%&*+=/\\", "()[]{}<>:;", "⇧'\"`~^|,⌫", "abc @ . space _ - ↵"],
    };
    const kb = document.createElement("div");
    kb.className = "okb"; kb.setAttribute("aria-hidden", "true");
    document.body.append(kb);
    let target = null, layer = "abc", shift = false;
    const typable = e => e && ((e.tagName === "INPUT" && /^(text|email|password|search|tel|number|url|)$/.test(e.type)) || e.tagName === "TEXTAREA");
    const put = text => {
      if (!target) return;
      const v = target.value, s = target.selectionStart ?? v.length, e = target.selectionEnd ?? v.length;
      try { target.setRangeText(text, s, e, "end"); } catch (_) { target.value = v + text; }
      target.dispatchEvent(new Event("input", { bubbles: true }));
    };
    const back = () => {
      if (!target) return;
      const v = target.value; let s = target.selectionStart ?? v.length, e = target.selectionEnd ?? v.length;
      if (s === e && s > 0) s -= 1;
      try { target.setRangeText("", s, e, "end"); } catch (_) { target.value = v.slice(0, -1); }
      target.dispatchEvent(new Event("input", { bubbles: true }));
    };
    const render = () => {
      kb.textContent = "";
      ROWS[layer].forEach(row => {
        const r = document.createElement("div"); r.className = "okb-row";
        const keys = row.includes(" ") ? row.split(" ") : [...row];
        keys.forEach(k => {
          const b = document.createElement("button"); b.type = "button"; b.dataset.k = k;
          b.textContent = k === "space" ? "" : (shift && k.length === 1 ? k.toUpperCase() : k);
          b.className = "okb-k" + ({ space: " wide", "⇧": " fn", "⌫": " fn", "↵": " fn go", "?123": " fn", abc: " fn" }[k] || "") + (k === "⇧" && shift ? " on" : "");
          r.append(b);
        });
        kb.append(r);
      });
    };
    kb.addEventListener("pointerdown", ev => ev.preventDefault());           // keep focus in the field
    kb.addEventListener("click", ev => {
      const k = ev.target.closest("button")?.dataset.k; if (!k) return;
      if (k === "⇧") { shift = !shift; render(); }
      else if (k === "⌫") back();
      else if (k === "?123") { layer = "sym"; render(); }
      else if (k === "abc") { layer = "abc"; render(); }
      else if (k === "space") put(" ");
      else if (k === "↵") { if (target?.tagName === "TEXTAREA") put("\n"); else target?.form?.requestSubmit(); }
      else { put(shift ? k.toUpperCase() : k); if (shift) { shift = false; render(); } }
    });
    document.addEventListener("focusin", ev => {
      if (!typable(ev.target)) return;
      target = ev.target; layer = target.type === "number" || target.type === "tel" ? "sym" : "abc"; render();
      kb.classList.add("on"); document.body.classList.add("okb-open");
      target.scrollIntoView({ block: "center" });
    });
    document.addEventListener("focusout", () => setTimeout(() => {
      if (!typable(document.activeElement)) { kb.classList.remove("on"); document.body.classList.remove("okb-open"); target = null; }
    }, 0));
  }

  if (document.body.classList.contains("kiosk")) {         // the device's own screen: back to device home when idle
    let idle;
    // a shared screen: when nobody has touched it for 3 minutes, sign out (the server then shows the device home)
    const leave = () => {
      const out = document.querySelector('form[action$="/logout"]');
      if (out) out.submit(); else location.href = "/device";
    };
    const arm = () => { clearTimeout(idle); idle = setTimeout(leave, 3 * 60 * 1000); };
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
})();
