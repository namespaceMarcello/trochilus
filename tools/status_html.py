#!/usr/bin/env python3
"""status_html.py -- la mappa del progetto, da docs/status.json a una pagina sola.

Il dato sta nel repo (docs/status.json): tappe, blocchi, stato di ognuno, i numeri misurati con
la data, le domande di docs/MEASUREMENTS.md collegate, i file e i comandi. Questo script ne fa una
pagina autosufficiente in build/stato/index.html, che si pubblica come artifact.

    tools/.venv/Scripts/python.exe tools/status_html.py [--out build/stato/index.html]

Quando cambia docs/STATUS.md cambia anche docs/status.json: il generatore non inventa niente.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "status.json"
GLOSSARIO = ROOT / "docs" / "glossary.json"

PAGE = """<title>Mappa di Trochilus</title>
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
    --fatto: #1f7361;
    --fatto-bg: #dceee9;
    --corso: #a85426;
    --corso-bg: #f7e5d8;
    --prossimo: #235f95;
    --prossimo-bg: #dbe8f4;
    --aperto: #6b7884;
    --aperto-bg: #e4e9ec;
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
      --fatto: #58b8a0;
      --fatto-bg: #12302a;
      --corso: #dd8b56;
      --corso-bg: #36241a;
      --prossimo: #74a9de;
      --prossimo-bg: #182839;
      --aperto: #8b98a3;
      --aperto-bg: #1e262d;
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
    --fatto: #58b8a0;
    --fatto-bg: #12302a;
    --corso: #dd8b56;
    --corso-bg: #36241a;
    --prossimo: #74a9de;
    --prossimo-bg: #182839;
    --aperto: #8b98a3;
    --aperto-bg: #1e262d;
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
  header.top .quando {
    margin-left: auto; font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums;
  }

  .legenda { display: flex; flex-wrap: wrap; gap: 8px; padding-block: 14px; align-items: center; }
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
  .chip.s-fatto { color: var(--fatto); }
  .chip.s-in-corso { color: var(--corso); }
  .chip.s-prossimo { color: var(--prossimo); }
  .chip.s-aperto { color: var(--aperto); }
  .chip.s-no { color: var(--no); }
  .legenda .nota { color: var(--muted); font-size: 13px; margin-left: 4px; }

  .rail-scroll { overflow-x: auto; overflow-y: hidden; padding-block: 8px 34px; }
  .rail { position: relative; display: flex; gap: 22px; align-items: flex-start; min-width: min-content; }
  svg.edges { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; overflow: visible; }
  svg.edges path { fill: none; stroke: var(--edge); stroke-width: 1.5; opacity: .75; }
  svg.edges path.on { stroke: var(--edge-on); stroke-width: 2.2; opacity: 1; }

  .colonna { position: relative; z-index: 1; width: 252px; flex: 0 0 252px; display: flex; flex-direction: column; gap: 10px; }
  .tappa-testa { display: flex; flex-direction: column; gap: 5px; padding-bottom: 4px; }
  .tappa-id {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; letter-spacing: .08em;
    color: var(--muted);
  }
  .tappa-titolo {
    font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif; font-weight: 700;
    font-size: 19px; line-height: 1.2; text-wrap: balance;
  }
  .tappa-sommario { color: var(--muted); font-size: 13px; }
  .barra { height: 4px; background: var(--surface-2); border-radius: 2px; overflow: hidden; }
  .barra > i { display: block; height: 100%; background: var(--fatto); }
  .barra-n {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .carta {
    position: relative; text-align: left; width: 100%; display: flex; flex-direction: column; gap: 6px;
    background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--aperto);
    border-radius: 7px; padding: 11px 12px; cursor: pointer; color: inherit;
    font: inherit; transition: transform .12s ease, box-shadow .12s ease;
  }
  .carta:hover { transform: translateY(-1px); box-shadow: var(--shadow); }
  .carta:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .carta[aria-current="true"] { box-shadow: var(--shadow); border-color: var(--accent); }
  .carta.s-fatto { border-left-color: var(--fatto); }
  .carta.s-in-corso { border-left-color: var(--corso); }
  .carta.s-prossimo { border-left-color: var(--prossimo); }
  .carta.s-aperto { border-left-color: var(--aperto); }
  .carta.s-no { border-left-color: var(--no); }
  .carta.spenta { opacity: .3; }
  .carta h3 { margin: 0; font-size: 15px; font-weight: 600; line-height: 1.3; }
  .carta .riga { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .stato {
    font-size: 11px; letter-spacing: .04em; text-transform: uppercase; font-weight: 600;
    border-radius: 4px; padding: 2px 6px;
  }
  .s-fatto .stato, .stato.s-fatto { background: var(--fatto-bg); color: var(--fatto); }
  .s-in-corso .stato, .stato.s-in-corso { background: var(--corso-bg); color: var(--corso); }
  .s-prossimo .stato, .stato.s-prossimo { background: var(--prossimo-bg); color: var(--prossimo); }
  .s-aperto .stato, .stato.s-aperto { background: var(--aperto-bg); color: var(--aperto); }
  .s-no .stato, .stato.s-no { background: var(--no-bg); color: var(--no); }
  .carta .primo {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px; color: var(--muted);
    line-height: 1.45; font-variant-numeric: tabular-nums;
  }
  .carta .dq { font-size: 11.5px; color: var(--muted); }

  .drawer {
    position: fixed; z-index: 20; right: 0; top: 0; bottom: 0; width: min(440px, 100%);
    background: var(--surface); border-left: 1px solid var(--line); box-shadow: var(--shadow);
    display: flex; flex-direction: column; transform: translateX(100%);
    transition: transform .18s ease; padding-top: env(safe-area-inset-top, 0px);
  }
  .drawer.aperto { transform: none; }
  .drawer-testa {
    display: flex; align-items: flex-start; gap: 12px; padding: 16px 16px 12px;
    border-bottom: 1px solid var(--line);
  }
  .drawer-testa h2 { margin: 0; font-size: 19px; line-height: 1.25; font-weight: 600; text-wrap: balance; }
  .drawer-testa .dove { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px; color: var(--muted); }
  .chiudi {
    margin-left: auto; background: var(--surface-2); border: 1px solid var(--line); color: var(--ink);
    border-radius: 6px; width: 30px; height: 30px; font-size: 16px; cursor: pointer; line-height: 1;
  }
  .drawer-corpo { overflow-y: auto; padding: 4px 16px 24px; display: flex; flex-direction: column; gap: 18px; }
  .sezione h4 {
    margin: 0 0 6px; font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
    color: var(--muted); font-weight: 600;
  }
  .sezione p { margin: 0; }
  .sezione ul { margin: 0; padding-left: 0; list-style: none; display: flex; flex-direction: column; gap: 7px; }
  .misura { display: flex; gap: 9px; align-items: baseline; }
  .misura .data {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; color: var(--muted);
    white-space: nowrap; font-variant-numeric: tabular-nums;
  }
  .misura .v { font-size: 13.5px; }
  .domanda { display: flex; gap: 9px; align-items: baseline; font-size: 13.5px; }
  .domanda .n {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; border-radius: 4px;
    padding: 1px 5px; white-space: nowrap;
  }
  .domanda .n.chiusa { background: var(--fatto-bg); color: var(--fatto); }
  .domanda .n.aperta { background: var(--aperto-bg); color: var(--aperto); }
  code, .mono {
    font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12.5px;
    background: var(--surface-2); border-radius: 4px; padding: 1px 5px;
  }
  .sezione .lista-mono { display: flex; flex-direction: column; gap: 5px; align-items: flex-start; }
  .lista-mono code { overflow-wrap: anywhere; }
  .collega {
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
  .pop .vedi { display: flex; flex-wrap: wrap; gap: 6px; padding-top: 2px; }
  .pop .vedi button {
    background: var(--surface-2); border: 1px solid var(--line); color: var(--ink); cursor: pointer;
    border-radius: 999px; padding: 3px 9px; font: inherit; font-size: 12px;
  }
  .pop .eti { font-size: 10.5px; letter-spacing: .09em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
  .glossario-btn {
    background: var(--surface); border: 1px solid var(--line); color: var(--ink); cursor: pointer;
    border-radius: 999px; padding: 5px 12px; font: inherit; font-size: 13px;
  }
  .voce-gloss { display: flex; flex-direction: column; gap: 3px; padding-bottom: 10px; border-bottom: 1px solid var(--line); }
  .voce-gloss h5 { margin: 0; font-size: 14.5px; font-weight: 600; }
  .voce-gloss p { margin: 0; font-size: 13.5px; }
  .velo {
    position: fixed; inset: 0; background: rgba(6, 10, 13, .38); z-index: 19; border: 0; padding: 0;
    opacity: 0; pointer-events: none; transition: opacity .18s ease;
  }
  .velo.aperto { opacity: 1; pointer-events: auto; }
  footer.fine { color: var(--muted); font-size: 12.5px; padding-block: 18px 30px; border-top: 1px solid var(--line); }

  @media (max-width: 760px) {
    .rail { flex-direction: column; min-width: 0; }
    .colonna { width: 100%; flex: 1 1 auto; }
    svg.edges { display: none; }
    .rail-scroll { overflow-x: visible; }
    .drawer { top: auto; height: 82%; width: 100%; border-left: 0; border-top: 1px solid var(--line);
      border-radius: 12px 12px 0 0; transform: translateY(100%); padding-bottom: env(safe-area-inset-bottom, 0px); }
    .drawer.aperto { transform: none; }
  }
  @media (prefers-reduced-motion: reduce) {
    .carta, .drawer, .velo { transition: none; }
  }
</style>

<div class="wrap">
  <header class="top">
    <h1>Mappa di Trochilus</h1>
    <div class="sub" id="sottotitolo"></div>
    <div class="quando" id="quando"></div>
  </header>
  <div class="legenda" id="legenda"></div>
</div>
<div class="pop" id="pop" role="dialog" aria-live="polite" hidden></div>
<div class="wrap rail-scroll">
  <div class="rail" id="rail">
    <svg class="edges" id="edges" aria-hidden="true"></svg>
  </div>
</div>
<div class="wrap">
  <footer class="fine">
    Un blocco alla volta: clicca per sapere cosa vuol dire, i numeri che abbiamo misurato e le
    domande aperte collegate. Le frecce dicono cosa aspetta cosa. La verità sta nel repo:
    <code>docs/status.json</code>, generata da <code>tools/status_html.py</code>.
  </footer>
</div>

<button class="velo" id="velo" aria-label="chiudi il pannello"></button>
<aside class="drawer" id="drawer" role="dialog" aria-modal="false" aria-labelledby="drawer-titolo" hidden>
  <div class="drawer-testa">
    <div>
      <div class="dove" id="drawer-dove"></div>
      <h2 id="drawer-titolo"></h2>
    </div>
    <button class="chiudi" id="chiudi" aria-label="chiudi">&times;</button>
  </div>
  <div class="drawer-corpo" id="drawer-corpo"></div>
</aside>

<script type="application/json" id="dati">__DATI__</script>
<script type="application/json" id="glossario">__GLOSSARIO__</script>
<script>
  (function () {
    var D = JSON.parse(document.getElementById("dati").textContent);
    var G = JSON.parse(document.getElementById("glossario").textContent);
    var perId = {};
    D.blocchi.forEach(function (b) { perId[b.id] = b; });
    var ETICHETTA = { "fatto": "fatto", "in-corso": "in corso", "prossimo": "prossimo", "aperto": "aperto", "no": "scartato" };
    var ORDINE = ["fatto", "in-corso", "prossimo", "aperto", "no"];
    var spenti = {};
    var scelto = null;

    document.getElementById("sottotitolo").textContent = D.sottotitolo;
    document.getElementById("quando").textContent = "aggiornato " + D.aggiornato;

    // legenda: ogni stato è anche un interruttore che spegne gli altri blocchi
    var legenda = document.getElementById("legenda");
    ORDINE.forEach(function (st) {
      var n = D.blocchi.filter(function (b) { return b.stato === st; }).length;
      if (!n) return;
      var b = document.createElement("button");
      b.className = "chip s-" + st;
      b.setAttribute("aria-pressed", "false");
      b.innerHTML = '<span class="pip"></span>' + ETICHETTA[st] + ' <span class="n">' + n + "</span>";
      b.title = D.stati[st] || "";
      b.addEventListener("click", function () {
        spenti[st] = !spenti[st];
        b.setAttribute("aria-pressed", spenti[st] ? "false" : "true");
        b.style.opacity = spenti[st] ? ".45" : "1";
        disegnaFiltro();
      });
      legenda.appendChild(b);
    });
    var nota = document.createElement("span");
    nota.className = "nota";
    nota.textContent = "clicca uno stato per metterlo da parte";
    legenda.appendChild(nota);

    var bottoneGloss = document.createElement("button");
    bottoneGloss.className = "glossario-btn";
    bottoneGloss.style.marginLeft = "auto";
    bottoneGloss.textContent = "Glossario (" + G.voci.length + " parole)";
    bottoneGloss.addEventListener("click", function () { apriGlossario(); });
    legenda.appendChild(bottoneGloss);

    // glossario: ogni parola chiave nel testo diventa cliccabile, la prima volta che compare
    var perTermine = {}, alias = [], idPerAlias = {};
    G.voci.forEach(function (v) {
      perTermine[v.id] = v;
      (v.alias || []).forEach(function (a) { alias.push(a); idPerAlias[a.toLowerCase()] = v.id; });
    });
    alias.sort(function (x, y) { return y.length - x.length; });
    function escRe(s) { return s.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&"); }
    var reGloss = new RegExp("(" + alias.map(escRe).join("|") + ")", "gi");
    var BORDO = /[0-9A-Za-zÀ-ÿ_]/;

    function esc(s) {
      return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function glossa(testo) {
      var visti = {}, out = "", last = 0, m;
      reGloss.lastIndex = 0;
      while ((m = reGloss.exec(testo)) !== null) {
        var off = m.index, fine = off + m[0].length;
        var prima = off > 0 ? testo.charAt(off - 1) : "";
        var dopo = testo.charAt(fine);
        var id = idPerAlias[m[0].toLowerCase()];
        if (!id || visti[id] || (prima && BORDO.test(prima)) || (dopo && BORDO.test(dopo))) continue;
        visti[id] = 1;
        out += esc(testo.slice(last, off)) +
          '<button class="gloss" data-g="' + id + '" title="che cos\\'è?">' + esc(m[0]) + "</button>";
        last = fine;
      }
      return out + esc(testo.slice(last));
    }

    var pop = document.getElementById("pop");
    var ancoraPop = null;

    function apriPop(id, ancora) {
      var v = perTermine[id];
      if (!v) return;
      ancoraPop = ancora;
      var vedi = (v.vedi || []).filter(function (x) { return perTermine[x]; });
      pop.innerHTML = '<div class="eti">glossario</div><h5>' + esc(v.titolo) + "</h5><p>" + esc(v.testo) + "</p>" +
        (vedi.length ? '<div class="vedi">' + vedi.map(function (x) {
          return '<button data-g2="' + x + '">' + esc(perTermine[x].titolo) + "</button>";
        }).join("") + "</div>" : "");
      pop.hidden = false;
      var r = ancora.getBoundingClientRect();
      var w = pop.offsetWidth, h = pop.offsetHeight;
      var x = Math.min(Math.max(12, r.left - 8), window.innerWidth - w - 12);
      var y = r.bottom + 8;
      if (y + h > window.innerHeight - 12) y = Math.max(12, r.top - h - 8);
      pop.style.left = x + "px";
      pop.style.top = y + "px";
      Array.prototype.forEach.call(pop.querySelectorAll("[data-g2]"), function (el) {
        el.addEventListener("click", function () { apriPop(el.dataset.g2, ancoraPop || ancora); });
      });
    }

    function chiudiPop() { pop.hidden = true; }

    document.addEventListener("click", function (e) {
      var g = e.target.closest ? e.target.closest(".gloss") : null;
      if (g) { apriPop(g.dataset.g, g); return; }
      if (!pop.hidden && !pop.contains(e.target)) chiudiPop();
    });
    window.addEventListener("resize", chiudiPop);

    // colonne e carte
    var rail = document.getElementById("rail");
    D.tappe.forEach(function (t) {
      var col = document.createElement("div");
      col.className = "colonna";
      var blocchi = D.blocchi.filter(function (b) { return b.tappa === t.id; });
      var fatti = blocchi.filter(function (b) { return b.stato === "fatto"; }).length;
      var testa = document.createElement("div");
      testa.className = "tappa-testa";
      var quota = blocchi.length ? Math.round(fatti / blocchi.length * 100) : 0;
      testa.innerHTML =
        '<div class="tappa-id">' + t.id + "</div>" +
        '<div class="tappa-titolo">' + t.titolo + "</div>" +
        '<div class="tappa-sommario">' + glossa(t.sommario) + "</div>" +
        '<div class="barra"><i style="width:' + quota + '%"></i></div>' +
        '<div class="barra-n">' + fatti + " di " + blocchi.length + " fatti</div>";
      col.appendChild(testa);
      blocchi.forEach(function (b) {
        var c = document.createElement("button");
        c.className = "carta s-" + b.stato;
        c.id = "carta-" + b.id;
        c.dataset.stato = b.stato;
        var primo = b.numeri && b.numeri.length ? '<div class="primo">' + b.numeri[0].v + "</div>" : "";
        var aperte = (b.domande || []).filter(function (d) { return d.stato === "aperta"; }).length;
        var dq = aperte ? '<div class="dq">' + aperte + (aperte === 1 ? " domanda aperta" : " domande aperte") + "</div>" : "";
        c.innerHTML =
          '<div class="riga"><span class="stato s-' + b.stato + '">' + ETICHETTA[b.stato] + "</span></div>" +
          "<h3>" + b.titolo + "</h3>" + primo + dq;
        c.addEventListener("click", function () { apri(b.id); });
        col.appendChild(c);
      });
      rail.appendChild(col);
    });

    // frecce: da ciò che serve a ciò che aspetta
    var svg = document.getElementById("edges");
    function disegnaFrecce() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      if (window.innerWidth <= 760) return;
      var base = rail.getBoundingClientRect();
      svg.setAttribute("viewBox", "0 0 " + rail.scrollWidth + " " + rail.scrollHeight);
      svg.setAttribute("width", rail.scrollWidth);
      svg.setAttribute("height", rail.scrollHeight);
      D.blocchi.forEach(function (b) {
        (b.dipende || []).forEach(function (da) {
          var a = document.getElementById("carta-" + da), z = document.getElementById("carta-" + b.id);
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
          p.dataset.da = da;
          p.dataset.a = b.id;
          svg.appendChild(p);
        });
      });
      evidenzia();
    }

    function evidenzia() {
      Array.prototype.forEach.call(svg.querySelectorAll("path"), function (p) {
        var on = scelto && (p.dataset.da === scelto || p.dataset.a === scelto);
        p.classList.toggle("on", !!on);
      });
    }

    function disegnaFiltro() {
      D.blocchi.forEach(function (b) {
        var c = document.getElementById("carta-" + b.id);
        if (c) c.classList.toggle("spenta", !!spenti[b.stato]);
      });
    }

    // pannello
    var drawer = document.getElementById("drawer"), velo = document.getElementById("velo");
    var corpo = document.getElementById("drawer-corpo");

    function sezione(titolo, dentro) {
      if (!dentro) return "";
      return '<div class="sezione"><h4>' + titolo + "</h4>" + dentro + "</div>";
    }

    function apri(id) {
      var b = perId[id];
      if (!b) return;
      var t = D.tappe.filter(function (x) { return x.id === b.tappa; })[0];
      scelto = id;
      document.getElementById("drawer-dove").textContent = b.tappa + " · " + (t ? t.titolo : "");
      document.getElementById("drawer-titolo").textContent = b.titolo;

      var html = "";
      html += sezione("stato", '<p><span class="stato s-' + b.stato + '">' + ETICHETTA[b.stato] + "</span> " +
        '<span style="color:var(--muted);font-size:13px">' + (D.stati[b.stato] || "") + "</span></p>");
      html += sezione("cosa vuol dire", "<p>" + glossa(b.cosa) + "</p>");
      if (b.numeri && b.numeri.length) {
        html += sezione("numeri misurati", "<ul>" + b.numeri.map(function (n) {
          return '<li class="misura"><span class="data">' + n.d + '</span><span class="v">' + glossa(n.v) + "</span></li>";
        }).join("") + "</ul>");
      }
      if (b.domande && b.domande.length) {
        html += sezione("domande collegate", "<ul>" + b.domande.map(function (d) {
          return '<li class="domanda"><span class="n ' + d.stato + '">#' + d.n + "</span><span>" + glossa(d.testo) + "</span></li>";
        }).join("") + "</ul>");
      }
      var dip = (b.dipende || []).filter(function (x) { return perId[x]; });
      if (dip.length) {
        html += sezione("aspetta", '<ul id="dip-lista">' + dip.map(function (x) {
          return '<li><button class="collega" data-va="' + x + '">' + perId[x].titolo +
            ' <span class="stato s-' + perId[x].stato + '">' + ETICHETTA[perId[x].stato] + "</span></button></li>";
        }).join("") + "</ul>");
      }
      var seguiti = D.blocchi.filter(function (x) { return (x.dipende || []).indexOf(id) >= 0; });
      if (seguiti.length) {
        html += sezione("sblocca", "<ul>" + seguiti.map(function (x) {
          return '<li><button class="collega" data-va="' + x.id + '">' + x.titolo +
            ' <span class="stato s-' + x.stato + '">' + ETICHETTA[x.stato] + "</span></button></li>";
        }).join("") + "</ul>");
      }
      if ((b.file && b.file.length) || (b.comandi && b.comandi.length)) {
        var mono = "";
        if (b.file && b.file.length) {
          mono += '<div class="lista-mono">' + b.file.map(function (f) { return "<code>" + f + "</code>"; }).join("") + "</div>";
        }
        if (b.comandi && b.comandi.length) {
          mono += '<div class="lista-mono" style="margin-top:6px">' + b.comandi.map(function (c) {
            return "<code>" + c + "</code>";
          }).join("") + "</div>";
        }
        html += sezione("dove sta, come si prova", mono);
      }
      corpo.innerHTML = html;
      Array.prototype.forEach.call(corpo.querySelectorAll("[data-va]"), function (el) {
        el.addEventListener("click", function () { apri(el.dataset.va); });
      });
      corpo.scrollTop = 0;
      drawer.hidden = false;
      requestAnimationFrame(function () { drawer.classList.add("aperto"); velo.classList.add("aperto"); });
      D.blocchi.forEach(function (x) {
        var c = document.getElementById("carta-" + x.id);
        if (c) c.setAttribute("aria-current", x.id === id ? "true" : "false");
      });
      evidenzia();
    }

    function apriGlossario() {
      scelto = null;
      chiudiPop();
      document.getElementById("drawer-dove").textContent = "tutte le parole, spiegate da zero";
      document.getElementById("drawer-titolo").textContent = "Glossario";
      var voci = G.voci.slice().sort(function (a, b) { return a.titolo.localeCompare(b.titolo, "it"); });
      corpo.innerHTML = voci.map(function (v) {
        return '<div class="voce-gloss"><h5>' + esc(v.titolo) + "</h5><p>" + esc(v.testo) + "</p></div>";
      }).join("");
      corpo.scrollTop = 0;
      drawer.hidden = false;
      requestAnimationFrame(function () { drawer.classList.add("aperto"); velo.classList.add("aperto"); });
      evidenzia();
    }

    function chiudi() {
      scelto = null;
      chiudiPop();
      drawer.classList.remove("aperto");
      velo.classList.remove("aperto");
      setTimeout(function () { if (!drawer.classList.contains("aperto")) drawer.hidden = true; }, 200);
      D.blocchi.forEach(function (x) {
        var c = document.getElementById("carta-" + x.id);
        if (c) c.setAttribute("aria-current", "false");
      });
      evidenzia();
    }

    document.getElementById("chiudi").addEventListener("click", chiudi);
    velo.addEventListener("click", chiudi);
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") chiudi(); });
    window.addEventListener("resize", disegnaFrecce);
    disegnaFrecce();
    // le frecce dipendono dalle misure del testo: ridisegna quando i font arrivano
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(disegnaFrecce);
    // il primo blocco che sta per essere lavorato, aperto: la pagina non si apre vuota
    var prossimo = D.blocchi.filter(function (b) { return b.stato === "prossimo" || b.stato === "in-corso"; })[0];
    if (prossimo && window.innerWidth > 760) apri(prossimo.id);
  })();
</script>
"""


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "stato" / "index.html"))
    args = ap.parse_args(argv[1:])

    data = json.loads(DATA.read_text(encoding="utf-8"))
    gloss = json.loads(GLOSSARIO.read_text(encoding="utf-8"))

    # il JSON finisce dentro un tag <script>: </script> dentro una stringa chiuderebbe il tag
    def blob(x):
        return json.dumps(x, ensure_ascii=False).replace("</", "<\\/")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pagina = PAGE.replace("__DATI__", blob(data)).replace("__GLOSSARIO__", blob(gloss))
    out.write_text(pagina, encoding="utf-8")
    n_blocchi = len(data["blocchi"])
    fatti = sum(1 for b in data["blocchi"] if b["stato"] == "fatto")
    print(f"stato_html: {out} ({out.stat().st_size // 1024} KiB, {fatti}/{n_blocchi} blocchi fatti, "
          f"{len(gloss['voci'])} voci di glossario)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
