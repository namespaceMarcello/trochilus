#!/usr/bin/env python3
"""status_html.py -- the project's map, from docs/status.json to a single page.

The data lives in the repo (docs/status.json): milestones, blocks, the state of each, the measured
numbers with their date, the linked questions of docs/MEASUREMENTS.md, the files and the commands.
This script turns it into a self-contained page in build/status/index.html, published as an artifact.

    tools/.venv/Scripts/python.exe tools/status_html.py [--out build/status/index.html]

When docs/STATUS.md changes, docs/status.json changes too: the generator invents nothing.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "status.json"
GLOSSARY = ROOT / "docs" / "glossary.json"

PAGE = """<title>Trochilus Map</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
  :root {
    --ground: #f2f4f5;
    --surface: #ffffff;
    --surface-2: #e9edef;
    --ink: #11171c;
    --muted: #5d6b76;
    --line: #d5dce0;
    --accent: #a85426;
    --done: #1f7361;
    --done-bg: #dceee9;
    --progress: #a85426;
    --progress-bg: #f7e5d8;
    --next: #235f95;
    --next-bg: #dbe8f4;
    --open: #6b7884;
    --open-bg: #e4e9ec;
    --no: #7d4646;
    --no-bg: #eedede;
    --edge: #b6c2c9;
    --edge-on: #a85426;
    --shadow: 0 1px 2px rgba(17, 23, 28, .06), 0 8px 24px rgba(17, 23, 28, .06);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground: #0d1115;
      --surface: #151b21;
      --surface-2: #1d252c;
      --ink: #e4eaee;
      --muted: #94a2ad;
      --line: #27313a;
      --accent: #dd8b56;
      --done: #58b8a0;
      --done-bg: #12302a;
      --progress: #dd8b56;
      --progress-bg: #36241a;
      --next: #74a9de;
      --next-bg: #182839;
      --open: #8b98a3;
      --open-bg: #1e262d;
      --no: #c58787;
      --no-bg: #2f1f1f;
      --edge: #38454f;
      --edge-on: #dd8b56;
      --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 10px 28px rgba(0, 0, 0, .35);
    }
  }
  :root[data-theme="dark"] {
    --ground: #0d1115;
    --surface: #151b21;
    --surface-2: #1d252c;
    --ink: #e4eaee;
    --muted: #94a2ad;
    --line: #27313a;
    --accent: #dd8b56;
    --done: #58b8a0;
    --done-bg: #12302a;
    --progress: #dd8b56;
    --progress-bg: #36241a;
    --next: #74a9de;
    --next-bg: #182839;
    --open: #8b98a3;
    --open-bg: #1e262d;
    --no: #c58787;
    --no-bg: #2f1f1f;
    --edge: #38454f;
    --edge-on: #dd8b56;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 10px 28px rgba(0, 0, 0, .35);
  }

  body {
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font: 400 15px/1.55 "IBM Plex Sans", system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { padding: 0 16px; }
  header.top {
    display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 18px;
    padding-block: 26px 14px; border-bottom: 1px solid var(--line);
  }
  header.top h1 {
    margin: 0; font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif;
    font-weight: 700; font-size: clamp(26px, 5vw, 38px); letter-spacing: -.01em;
  }
  header.top .sub { color: var(--muted); font-size: 14px; max-width: 62ch; }
  header.top .when {
    margin-left: auto; font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums;
  }

  .legend { display: flex; flex-wrap: wrap; gap: 8px; padding-block: 14px; align-items: center; }
  .chip {
    display: inline-flex; align-items: center; gap: 7px; border: 1px solid var(--line);
    background: var(--surface); color: var(--ink); border-radius: 999px;
    padding: 5px 11px 5px 9px; font-size: 13px; cursor: pointer;
  }
  .chip[aria-pressed="true"] { border-color: currentColor; }
  .chip .n {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px;
    color: var(--muted); font-variant-numeric: tabular-nums;
  }
  .chip .pip { width: 9px; height: 9px; border-radius: 50%; background: currentColor; }
  .chip.s-done { color: var(--done); }
  .chip.s-in-progress { color: var(--progress); }
  .chip.s-next { color: var(--next); }
  .chip.s-open { color: var(--open); }
  .chip.s-no { color: var(--no); }
  .legend .note { color: var(--muted); font-size: 13px; margin-left: 4px; }

  .rail-scroll { overflow-x: auto; overflow-y: hidden; padding-block: 8px 34px; }
  .rail { position: relative; display: flex; gap: 22px; align-items: flex-start; min-width: min-content; }
  svg.edges { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; overflow: visible; }
  svg.edges path { fill: none; stroke: var(--edge); stroke-width: 1.5; opacity: .75; }
  svg.edges path.on { stroke: var(--edge-on); stroke-width: 2.2; opacity: 1; }

  .column { position: relative; z-index: 1; width: 252px; flex: 0 0 252px; display: flex; flex-direction: column; gap: 10px; }
  .milestone-head { display: flex; flex-direction: column; gap: 5px; padding-bottom: 4px; }
  .milestone-id {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; letter-spacing: .08em;
    color: var(--muted);
  }
  .milestone-title {
    font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif; font-weight: 700;
    font-size: 19px; line-height: 1.2; text-wrap: balance;
  }
  .milestone-summary { color: var(--muted); font-size: 13px; }
  .bar { height: 4px; background: var(--surface-2); border-radius: 2px; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: var(--done); }
  .bar-n {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .card {
    position: relative; text-align: left; width: 100%; display: flex; flex-direction: column; gap: 6px;
    background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--open);
    border-radius: 7px; padding: 11px 12px; cursor: pointer; color: inherit;
    font: inherit; transition: transform .12s ease, box-shadow .12s ease;
  }
  .card:hover { transform: translateY(-1px); box-shadow: var(--shadow); }
  .card:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .card[aria-current="true"] { box-shadow: var(--shadow); border-color: var(--accent); }
  .card.s-done { border-left-color: var(--done); }
  .card.s-in-progress { border-left-color: var(--progress); }
  .card.s-next { border-left-color: var(--next); }
  .card.s-open { border-left-color: var(--open); }
  .card.s-no { border-left-color: var(--no); }
  .card.dimmed { opacity: .3; }
  .card h3 { margin: 0; font-size: 15px; font-weight: 600; line-height: 1.3; }
  .card .row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .state {
    font-size: 11px; letter-spacing: .04em; text-transform: uppercase; font-weight: 600;
    border-radius: 4px; padding: 2px 6px;
  }
  .s-done .state, .state.s-done { background: var(--done-bg); color: var(--done); }
  .s-in-progress .state, .state.s-in-progress { background: var(--progress-bg); color: var(--progress); }
  .s-next .state, .state.s-next { background: var(--next-bg); color: var(--next); }
  .s-open .state, .state.s-open { background: var(--open-bg); color: var(--open); }
  .s-no .state, .state.s-no { background: var(--no-bg); color: var(--no); }
  .card .first {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px; color: var(--muted);
    line-height: 1.45; font-variant-numeric: tabular-nums;
  }
  .card .dq { font-size: 11.5px; color: var(--muted); }

  .drawer {
    position: fixed; z-index: 20; right: 0; top: 0; bottom: 0; width: min(440px, 100%);
    background: var(--surface); border-left: 1px solid var(--line); box-shadow: var(--shadow);
    display: flex; flex-direction: column; transform: translateX(100%);
    transition: transform .18s ease; padding-top: env(safe-area-inset-top, 0px);
  }
  .drawer.open { transform: none; }
  .drawer-head {
    display: flex; align-items: flex-start; gap: 12px; padding: 16px 16px 12px;
    border-bottom: 1px solid var(--line);
  }
  .drawer-head h2 { margin: 0; font-size: 19px; line-height: 1.25; font-weight: 600; text-wrap: balance; }
  .drawer-head .where { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px; color: var(--muted); }
  .close {
    margin-left: auto; background: var(--surface-2); border: 1px solid var(--line); color: var(--ink);
    border-radius: 6px; width: 30px; height: 30px; font-size: 16px; cursor: pointer; line-height: 1;
  }
  .drawer-body { overflow-y: auto; padding: 4px 16px 24px; display: flex; flex-direction: column; gap: 18px; }
  .section h4 {
    margin: 0 0 6px; font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
    color: var(--muted); font-weight: 600;
  }
  .section p { margin: 0; }
  .section ul { margin: 0; padding-left: 0; list-style: none; display: flex; flex-direction: column; gap: 7px; }
  .measure { display: flex; gap: 9px; align-items: baseline; }
  .measure .date {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; color: var(--muted);
    white-space: nowrap; font-variant-numeric: tabular-nums;
  }
  .measure .v { font-size: 13.5px; }
  .question { display: flex; gap: 9px; align-items: baseline; font-size: 13.5px; }
  .question .n {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; border-radius: 4px;
    padding: 1px 5px; white-space: nowrap;
  }
  .question .n.closed { background: var(--done-bg); color: var(--done); }
  .question .n.open { background: var(--open-bg); color: var(--open); }
  code, .mono {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12.5px;
    background: var(--surface-2); border-radius: 4px; padding: 1px 5px;
  }
  .section .mono-list { display: flex; flex-direction: column; gap: 5px; align-items: flex-start; }
  .mono-list code { overflow-wrap: anywhere; }
  .link {
    background: none; border: 0; padding: 0; color: var(--accent); cursor: pointer;
    font: inherit; font-size: 13.5px; text-align: left; text-decoration: underline;
    text-underline-offset: 2px;
  }
  .gloss {
    background: none; border: 0; padding: 0; margin: 0; font: inherit; color: inherit;
    cursor: help; text-decoration: underline dotted; text-decoration-color: var(--accent);
    text-underline-offset: 3px; text-decoration-thickness: 1.5px;
  }
  .gloss:hover { background: var(--surface-2); border-radius: 3px; }
  .gloss:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 3px; }
  .pop {
    position: fixed; z-index: 30; width: min(330px, calc(100vw - 32px)); background: var(--surface);
    border: 1px solid var(--line); border-radius: 9px; box-shadow: var(--shadow); padding: 13px 14px;
    display: flex; flex-direction: column; gap: 8px;
  }
  .pop h5 { margin: 0; font-size: 14.5px; font-weight: 600; }
  .pop p { margin: 0; font-size: 13.5px; line-height: 1.5; }
  .pop .see { display: flex; flex-wrap: wrap; gap: 6px; padding-top: 2px; }
  .pop .see button {
    background: var(--surface-2); border: 1px solid var(--line); color: var(--ink); cursor: pointer;
    border-radius: 999px; padding: 3px 9px; font: inherit; font-size: 12px;
  }
  .pop .tag { font-size: 10.5px; letter-spacing: .09em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
  .glossary-btn {
    background: var(--surface); border: 1px solid var(--line); color: var(--ink); cursor: pointer;
    border-radius: 999px; padding: 5px 12px; font: inherit; font-size: 13px;
  }
  .gloss-entry { display: flex; flex-direction: column; gap: 3px; padding-bottom: 10px; border-bottom: 1px solid var(--line); }
  .gloss-entry h5 { margin: 0; font-size: 14.5px; font-weight: 600; }
  .gloss-entry p { margin: 0; font-size: 13.5px; }
  .veil {
    position: fixed; inset: 0; background: rgba(6, 10, 13, .38); z-index: 19; border: 0; padding: 0;
    opacity: 0; pointer-events: none; transition: opacity .18s ease;
  }
  .veil.open { opacity: 1; pointer-events: auto; }
  footer.end { color: var(--muted); font-size: 12.5px; padding-block: 18px 30px; border-top: 1px solid var(--line); }

  @media (max-width: 760px) {
    .rail { flex-direction: column; min-width: 0; }
    .column { width: 100%; flex: 1 1 auto; }
    svg.edges { display: none; }
    .rail-scroll { overflow-x: visible; }
    .drawer { top: auto; height: 82%; width: 100%; border-left: 0; border-top: 1px solid var(--line);
      border-radius: 12px 12px 0 0; transform: translateY(100%); padding-bottom: env(safe-area-inset-bottom, 0px); }
    .drawer.open { transform: none; }
  }
  @media (prefers-reduced-motion: reduce) {
    .card, .drawer, .veil { transition: none; }
  }
</style>

<div class="wrap">
  <header class="top">
    <h1>Trochilus Map</h1>
    <div class="sub" id="subtitle"></div>
    <div class="when" id="when"></div>
  </header>
  <div class="legend" id="legend"></div>
</div>
<div class="pop" id="pop" role="dialog" aria-live="polite" hidden></div>
<div class="wrap rail-scroll">
  <div class="rail" id="rail">
    <svg class="edges" id="edges" aria-hidden="true"></svg>
  </div>
</div>
<div class="wrap">
  <footer class="end">
    One block at a time: click one to learn what it means, the numbers we measured and the linked
    open questions. The arrows say what waits for what. The truth lives in the repo:
    <code>docs/status.json</code>, turned into this page by <code>tools/status_html.py</code>.
  </footer>
</div>

<button class="veil" id="veil" aria-label="close the panel"></button>
<aside class="drawer" id="drawer" role="dialog" aria-modal="false" aria-labelledby="drawer-title" hidden>
  <div class="drawer-head">
    <div>
      <div class="where" id="drawer-where"></div>
      <h2 id="drawer-title"></h2>
    </div>
    <button class="close" id="close" aria-label="close">&times;</button>
  </div>
  <div class="drawer-body" id="drawer-body"></div>
</aside>

<script type="application/json" id="data">__DATA__</script>
<script type="application/json" id="glossary">__GLOSSARY__</script>
<script>
  (function () {
    var D = JSON.parse(document.getElementById("data").textContent);
    var G = JSON.parse(document.getElementById("glossary").textContent);
    var byId = {};
    D.blocks.forEach(function (b) { byId[b.id] = b; });
    var LABEL = { "done": "done", "in-progress": "in progress", "next": "next", "open": "open", "no": "rejected" };
    var ORDER = ["done", "in-progress", "next", "open", "no"];
    var hidden = {};
    var chosen = null;

    document.getElementById("subtitle").textContent = D.subtitle;
    document.getElementById("when").textContent = "updated " + D.updated;

    // legend: every state is also a switch that dims the other blocks
    var legend = document.getElementById("legend");
    ORDER.forEach(function (st) {
      var n = D.blocks.filter(function (b) { return b.state === st; }).length;
      if (!n) return;
      var b = document.createElement("button");
      b.className = "chip s-" + st;
      b.setAttribute("aria-pressed", "false");
      b.innerHTML = '<span class="pip"></span>' + LABEL[st] + ' <span class="n">' + n + "</span>";
      b.title = D.states[st] || "";
      b.addEventListener("click", function () {
        hidden[st] = !hidden[st];
        b.setAttribute("aria-pressed", hidden[st] ? "false" : "true");
        b.style.opacity = hidden[st] ? ".45" : "1";
        drawFilter();
      });
      legend.appendChild(b);
    });
    var note = document.createElement("span");
    note.className = "note";
    note.textContent = "click a state to set it aside";
    legend.appendChild(note);

    var glossButton = document.createElement("button");
    glossButton.className = "glossary-btn";
    glossButton.style.marginLeft = "auto";
    glossButton.textContent = "Glossary (" + G.entries.length + " words)";
    glossButton.addEventListener("click", function () { openGlossary(); });
    legend.appendChild(glossButton);

    // glossary: every key word in the text becomes clickable, the first time it appears
    var byTerm = {}, alias = [], idByAlias = {};
    G.entries.forEach(function (v) {
      byTerm[v.id] = v;
      (v.alias || []).forEach(function (a) { alias.push(a); idByAlias[a.toLowerCase()] = v.id; });
    });
    alias.sort(function (x, y) { return y.length - x.length; });
    function escRe(s) { return s.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&"); }
    var reGloss = new RegExp("(" + alias.map(escRe).join("|") + ")", "gi");
    var EDGE = /[0-9A-Za-zÀ-ÿ_]/;

    function esc(s) {
      return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function gloss(text) {
      var seen = {}, out = "", last = 0, m;
      reGloss.lastIndex = 0;
      while ((m = reGloss.exec(text)) !== null) {
        var off = m.index, end = off + m[0].length;
        var before = off > 0 ? text.charAt(off - 1) : "";
        var after = text.charAt(end);
        var id = idByAlias[m[0].toLowerCase()];
        if (!id || seen[id] || (before && EDGE.test(before)) || (after && EDGE.test(after))) continue;
        seen[id] = 1;
        out += esc(text.slice(last, off)) +
          '<button class="gloss" data-g="' + id + '" title="what is it?">' + esc(m[0]) + "</button>";
        last = end;
      }
      return out + esc(text.slice(last));
    }

    var pop = document.getElementById("pop");
    var popAnchor = null;

    function openPop(id, anchor) {
      var v = byTerm[id];
      if (!v) return;
      popAnchor = anchor;
      var see = (v.see || []).filter(function (x) { return byTerm[x]; });
      pop.innerHTML = '<div class="tag">glossary</div><h5>' + esc(v.title) + "</h5><p>" + esc(v.text) + "</p>" +
        (see.length ? '<div class="see">' + see.map(function (x) {
          return '<button data-g2="' + x + '">' + esc(byTerm[x].title) + "</button>";
        }).join("") + "</div>" : "");
      pop.hidden = false;
      var r = anchor.getBoundingClientRect();
      var w = pop.offsetWidth, h = pop.offsetHeight;
      var x = Math.min(Math.max(12, r.left - 8), window.innerWidth - w - 12);
      var y = r.bottom + 8;
      if (y + h > window.innerHeight - 12) y = Math.max(12, r.top - h - 8);
      pop.style.left = x + "px";
      pop.style.top = y + "px";
      Array.prototype.forEach.call(pop.querySelectorAll("[data-g2]"), function (el) {
        el.addEventListener("click", function () { openPop(el.dataset.g2, popAnchor || anchor); });
      });
    }

    function closePop() { pop.hidden = true; }

    document.addEventListener("click", function (e) {
      var g = e.target.closest ? e.target.closest(".gloss") : null;
      if (g) { openPop(g.dataset.g, g); return; }
      if (!pop.hidden && !pop.contains(e.target)) closePop();
    });
    window.addEventListener("resize", closePop);

    // columns and cards
    var rail = document.getElementById("rail");
    D.milestones.forEach(function (t) {
      var col = document.createElement("div");
      col.className = "column";
      var blocks = D.blocks.filter(function (b) { return b.milestone === t.id; });
      var done = blocks.filter(function (b) { return b.state === "done"; }).length;
      var head = document.createElement("div");
      head.className = "milestone-head";
      var share = blocks.length ? Math.round(done / blocks.length * 100) : 0;
      head.innerHTML =
        '<div class="milestone-id">' + t.id + "</div>" +
        '<div class="milestone-title">' + t.title + "</div>" +
        '<div class="milestone-summary">' + gloss(t.summary) + "</div>" +
        '<div class="bar"><i style="width:' + share + '%"></i></div>' +
        '<div class="bar-n">' + done + " of " + blocks.length + " done</div>";
      col.appendChild(head);
      blocks.forEach(function (b) {
        var c = document.createElement("button");
        c.className = "card s-" + b.state;
        c.id = "card-" + b.id;
        c.dataset.state = b.state;
        var first = b.numbers && b.numbers.length ? '<div class="first">' + b.numbers[0].v + "</div>" : "";
        var open = (b.questions || []).filter(function (d) { return d.state === "open"; }).length;
        var dq = open ? '<div class="dq">' + open + (open === 1 ? " open question" : " open questions") + "</div>" : "";
        c.innerHTML =
          '<div class="row"><span class="state s-' + b.state + '">' + LABEL[b.state] + "</span></div>" +
          "<h3>" + b.title + "</h3>" + first + dq;
        c.addEventListener("click", function () { openBlock(b.id); });
        col.appendChild(c);
      });
      rail.appendChild(col);
    });

    // arrows: from what is needed to what waits for it
    var svg = document.getElementById("edges");
    function drawArrows() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      if (window.innerWidth <= 760) return;
      var base = rail.getBoundingClientRect();
      svg.setAttribute("viewBox", "0 0 " + rail.scrollWidth + " " + rail.scrollHeight);
      svg.setAttribute("width", rail.scrollWidth);
      svg.setAttribute("height", rail.scrollHeight);
      D.blocks.forEach(function (b) {
        (b.depends || []).forEach(function (from) {
          var a = document.getElementById("card-" + from), z = document.getElementById("card-" + b.id);
          if (!a || !z) return;
          var ra = a.getBoundingClientRect(), rz = z.getBoundingClientRect();
          var x1 = ra.right - base.left, y1 = ra.top + ra.height / 2 - base.top;
          var x2 = rz.left - base.left, y2 = rz.top + rz.height / 2 - base.top;
          var d;
          if (x2 - x1 > 6) {
            var dx = Math.max(26, (x2 - x1) / 2);
            d = "M" + x1 + " " + y1 + " C" + (x1 + dx) + " " + y1 + " " + (x2 - dx) + " " + y2 + " " + x2 + " " + y2;
          } else {
            var xa = ra.left + ra.width / 2 - base.left, ya = ra.bottom - base.top;
            var xz = rz.left + rz.width / 2 - base.left, yz = rz.top - base.top;
            var m = (ya + yz) / 2;
            d = "M" + xa + " " + ya + " C" + xa + " " + m + " " + xz + " " + m + " " + xz + " " + yz;
          }
          var p = document.createElementNS("http://www.w3.org/2000/svg", "path");
          p.setAttribute("d", d);
          p.dataset.from = from;
          p.dataset.to = b.id;
          svg.appendChild(p);
        });
      });
      highlight();
    }

    function highlight() {
      Array.prototype.forEach.call(svg.querySelectorAll("path"), function (p) {
        var on = chosen && (p.dataset.from === chosen || p.dataset.to === chosen);
        p.classList.toggle("on", !!on);
      });
    }

    function drawFilter() {
      D.blocks.forEach(function (b) {
        var c = document.getElementById("card-" + b.id);
        if (c) c.classList.toggle("dimmed", !!hidden[b.state]);
      });
    }

    // the panel
    var drawer = document.getElementById("drawer"), veil = document.getElementById("veil");
    var body = document.getElementById("drawer-body");

    function section(title, inside) {
      if (!inside) return "";
      return '<div class="section"><h4>' + title + "</h4>" + inside + "</div>";
    }

    function openBlock(id) {
      var b = byId[id];
      if (!b) return;
      var t = D.milestones.filter(function (x) { return x.id === b.milestone; })[0];
      chosen = id;
      document.getElementById("drawer-where").textContent = b.milestone + " · " + (t ? t.title : "");
      document.getElementById("drawer-title").textContent = b.title;

      var html = "";
      html += section("state", '<p><span class="state s-' + b.state + '">' + LABEL[b.state] + "</span> " +
        '<span style="color:var(--muted);font-size:13px">' + (D.states[b.state] || "") + "</span></p>");
      html += section("what it means", "<p>" + gloss(b.what) + "</p>");
      if (b.numbers && b.numbers.length) {
        html += section("measured numbers", "<ul>" + b.numbers.map(function (n) {
          return '<li class="measure"><span class="date">' + n.d + '</span><span class="v">' + gloss(n.v) + "</span></li>";
        }).join("") + "</ul>");
      }
      if (b.questions && b.questions.length) {
        html += section("linked questions", "<ul>" + b.questions.map(function (d) {
          var n = d.n != null ? "#" + d.n : "·";
          return '<li class="question"><span class="n ' + d.state + '">' + n + "</span><span>" + gloss(d.text) + "</span></li>";
        }).join("") + "</ul>");
      }
      var dep = (b.depends || []).filter(function (x) { return byId[x]; });
      if (dep.length) {
        html += section("waits for", '<ul id="dep-list">' + dep.map(function (x) {
          return '<li><button class="link" data-go="' + x + '">' + byId[x].title +
            ' <span class="state s-' + byId[x].state + '">' + LABEL[byId[x].state] + "</span></button></li>";
        }).join("") + "</ul>");
      }
      var followers = D.blocks.filter(function (x) { return (x.depends || []).indexOf(id) >= 0; });
      if (followers.length) {
        html += section("unblocks", "<ul>" + followers.map(function (x) {
          return '<li><button class="link" data-go="' + x.id + '">' + x.title +
            ' <span class="state s-' + x.state + '">' + LABEL[x.state] + "</span></button></li>";
        }).join("") + "</ul>");
      }
      if ((b.files && b.files.length) || (b.commands && b.commands.length)) {
        var mono = "";
        if (b.files && b.files.length) {
          mono += '<div class="mono-list">' + b.files.map(function (f) { return "<code>" + f + "</code>"; }).join("") + "</div>";
        }
        if (b.commands && b.commands.length) {
          mono += '<div class="mono-list" style="margin-top:6px">' + b.commands.map(function (c) {
            return "<code>" + c + "</code>";
          }).join("") + "</div>";
        }
        html += section("where it lives, how to try it", mono);
      }
      body.innerHTML = html;
      Array.prototype.forEach.call(body.querySelectorAll("[data-go]"), function (el) {
        el.addEventListener("click", function () { openBlock(el.dataset.go); });
      });
      body.scrollTop = 0;
      drawer.hidden = false;
      requestAnimationFrame(function () { drawer.classList.add("open"); veil.classList.add("open"); });
      D.blocks.forEach(function (x) {
        var c = document.getElementById("card-" + x.id);
        if (c) c.setAttribute("aria-current", x.id === id ? "true" : "false");
      });
      highlight();
    }

    function openGlossary() {
      chosen = null;
      closePop();
      document.getElementById("drawer-where").textContent = "every word, explained from scratch";
      document.getElementById("drawer-title").textContent = "Glossary";
      var entries = G.entries.slice().sort(function (a, b) { return a.title.localeCompare(b.title, "en"); });
      body.innerHTML = entries.map(function (v) {
        return '<div class="gloss-entry"><h5>' + esc(v.title) + "</h5><p>" + esc(v.text) + "</p></div>";
      }).join("");
      body.scrollTop = 0;
      drawer.hidden = false;
      requestAnimationFrame(function () { drawer.classList.add("open"); veil.classList.add("open"); });
      highlight();
    }

    function closePanel() {
      chosen = null;
      closePop();
      drawer.classList.remove("open");
      veil.classList.remove("open");
      setTimeout(function () { if (!drawer.classList.contains("open")) drawer.hidden = true; }, 200);
      D.blocks.forEach(function (x) {
        var c = document.getElementById("card-" + x.id);
        if (c) c.setAttribute("aria-current", "false");
      });
      highlight();
    }

    document.getElementById("close").addEventListener("click", closePanel);
    veil.addEventListener("click", closePanel);
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") closePanel(); });
    window.addEventListener("resize", drawArrows);
    drawArrows();
    // the arrows depend on the text's measurements: draw them again when the fonts arrive
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(drawArrows);
    // the first block about to be worked on, opened: the page does not open empty
    var upcoming = D.blocks.filter(function (b) { return b.state === "next" || b.state === "in-progress"; })[0];
    if (upcoming && window.innerWidth > 760) openBlock(upcoming.id);
  })();
</script>
"""


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "status" / "index.html"))
    args = ap.parse_args(argv[1:])

    data = json.loads(DATA.read_text(encoding="utf-8"))
    gloss = json.loads(GLOSSARY.read_text(encoding="utf-8"))

    # the JSON ends up inside a <script> tag: a </script> inside a string would close the tag
    def blob(x):
        return json.dumps(x, ensure_ascii=False).replace("</", "<\\/")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    page = PAGE.replace("__DATA__", blob(data)).replace("__GLOSSARY__", blob(gloss))
    out.write_text(page, encoding="utf-8")
    n_blocks = len(data["blocks"])
    done = sum(1 for b in data["blocks"] if b["state"] == "done")
    print(f"status_html: {out} ({out.stat().st_size // 1024} KiB, {done}/{n_blocks} blocks done, "
          f"{len(gloss['entries'])} glossary entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
