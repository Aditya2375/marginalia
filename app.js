import * as pdfjs from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.min.mjs";
pdfjs.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.0.379/build/pdf.worker.min.mjs";

const $ = (s) => document.querySelector(s);
const MODEL = "Xenova/all-MiniLM-L6-v2";
const CHUNK_WORDS = 110, OVERLAP = 25;
const MIN_SCORE = 0.30;       // below this the best passage is not treated as an answer
const STOP = new Set("a an the of and or to in on for with is are was were be been it its this that these those as at by from what which who whom how why when where do does did can could should would will about into than then so if not no yes you your i we they he she them his her our".split(" "));

let extractor = null, pdfDoc = null, state = null;

function say(msg) { $("#status").textContent = msg; }
function step(name, pct) {
  const order = ["read", "chunk", "embed", "ready"];
  order.forEach((s, i) => {
    const li = document.querySelector(`[data-s=${s}]`);
    li.className = i < order.indexOf(name) ? "done" : i === order.indexOf(name) ? "on" : "";
  });
  if (name === "ready") order.forEach((s) => (document.querySelector(`[data-s=${s}]`).className = "done"));
  if (pct != null) $("#bar").style.width = pct + "%";
}
function fail(msg) { const e = $("#err"); e.textContent = msg; e.hidden = false; }
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const words = (s) => s.toLowerCase().match(/[a-z0-9][a-z0-9'-]*/g) || [];
const stem = (w) => w.replace(/(ing|ed|es|s)$/, "");
const keyWords = (s) => words(s).filter((w) => !STOP.has(w) && w.length > 2);

/* ---------- read ---------- */
async function readPdf(file) {
  const buf = await file.arrayBuffer();
  pdfDoc = await pdfjs.getDocument({ data: buf.slice(0) }).promise;
  const pages = [];
  for (let p = 1; p <= pdfDoc.numPages; p++) {
    const page = await pdfDoc.getPage(p);
    const tc = await page.getTextContent();
    let text = "", lastY = null;
    for (const it of tc.items) {
      if (lastY !== null && Math.abs(it.transform[5] - lastY) > 2) text += " ";
      text += it.str + (it.hasEOL ? " " : "");
      lastY = it.transform[5];
    }
    pages.push({ page: p, text: text.replace(/\s+/g, " ").trim() });
    step("read", (p / pdfDoc.numPages) * 25);
    say(`Reading page ${p} of ${pdfDoc.numPages}`);
  }
  return pages;
}
async function readText(file) {
  pdfDoc = null;
  const t = (await file.text()).replace(/\r/g, "");
  const parts = t.split(/\n{2,}/), pages = [];
  let cur = "", n = 1;
  for (const p of parts) { if ((cur + p).split(/\s+/).length > 350 && cur) { pages.push({ page: n++, text: cur.replace(/\s+/g, " ").trim() }); cur = ""; } cur += p + " "; }
  if (cur.trim()) pages.push({ page: n, text: cur.replace(/\s+/g, " ").trim() });
  return pages;
}

/* ---------- split (never crosses a page) ---------- */
function sentences(text) { return (text.match(/[^.!?]+(?:[.!?]+["')\]]*|$)/g) || [text]).map((s) => s.trim()).filter((s) => s.length > 1); }
function chunkPages(pages) {
  const chunks = [];
  for (const pg of pages) {
    if (!pg.text) continue;
    const sents = sentences(pg.text);
    let buf = [], count = 0;
    const flush = () => { if (buf.length) chunks.push({ page: pg.page, text: buf.join(" "), sents: buf.slice() }); };
    for (const s of sents) {
      const n = s.split(/\s+/).length;
      if (count + n > CHUNK_WORDS && buf.length) {
        flush();
        let keep = [], k = 0;
        for (let i = buf.length - 1; i >= 0 && k < OVERLAP; i--) { keep.unshift(buf[i]); k += buf[i].split(/\s+/).length; }
        buf = keep; count = k;
      }
      buf.push(s); count += n;
    }
    flush();
  }
  return chunks;
}

/* ---------- embed ---------- */
async function getExtractor() {
  if (extractor) return extractor;
  say("Loading the language model (first time only, about 25 MB)");
  const tf = await import("https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2");
  tf.env.allowLocalModels = false;
  extractor = await tf.pipeline("feature-extraction", MODEL, {
    progress_callback: (p) => { if (p.status === "progress" && p.total) say(`Downloading model ${Math.round((p.loaded / p.total) * 100)}%`); },
  });
  return extractor;
}
async function embed(texts) {
  const ex = await getExtractor();
  const out = await ex(texts, { pooling: "mean", normalize: true });
  const d = out.dims[1], arr = out.data, vecs = [];
  for (let i = 0; i < texts.length; i++) vecs.push(arr.slice(i * d, (i + 1) * d));
  return vecs;
}
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

/* ---------- ask ---------- */
function lexical(q, text) {
  const qs = new Set(keyWords(q).map(stem)); if (!qs.size) return 0;
  const ts = new Set(words(text).map(stem)); let hit = 0;
  qs.forEach((w) => { if (ts.has(w)) hit++; });
  return hit / qs.size;
}
async function answer(q) {
  const [qv] = await embed([q]);
  const scored = state.chunks.map((c, i) => {
    const sem = dot(qv, state.vecs[i]), lex = lexical(q, c.text);
    return { c, i, sem, lex, score: sem * 0.7 + lex * 0.3 };
  }).sort((a, b) => b.score - a.score);
  const top = scored.slice(0, 4);
  const best = top[0];
  const bestSem = Math.max(...top.map((t) => t.sem)), bestLex = Math.max(...top.map((t) => t.lex));
  const base = { qv, top, best };
  // gate: strong keyword overlap, or clearly similar meaning. Otherwise refuse.
  if (!best || !(bestLex >= 0.5 || bestSem >= 0.36)) return { ...base, found: false };
  const cand = [];
  for (const t of top) for (const s of t.c.sents) if (s.split(/\s+/).length >= 4) cand.push({ s, page: t.c.page, chunk: t.c });
  const sv = await embed(cand.map((x) => x.s));
  cand.forEach((x, i) => { x.sem = dot(qv, sv[i]); x.lex = lexical(q, x.s); x.score = x.sem * 0.6 + x.lex * 0.4; });
  cand.sort((a, b) => b.score - a.score);
  const lead = cand[0];
  if (!lead || (lead.lex < 0.34 && lead.sem < 0.33)) return { ...base, found: false };
  const pick = [];
  for (const x of cand) {
    if (pick.length >= 3) break;
    if (x.score < lead.score * 0.82 || pick.some((p) => p.s === x.s)) continue;
    pick.push(x);
  }
  pick.sort((a, b) => a.page - b.page || state.chunks.indexOf(a.chunk) - state.chunks.indexOf(b.chunk));
  return { ...base, found: true, pick };
}

/* ---------- semantic map (real 2-D projection of the passage vectors) ---------- */
function pca2(vecs) {
  const n = vecs.length, d = vecs[0].length, mean = new Float32Array(d);
  vecs.forEach((v) => { for (let j = 0; j < d; j++) mean[j] += v[j] / n; });
  const X = vecs.map((v) => Float32Array.from(v, (x, j) => x - mean[j]));
  const comps = [];
  for (let c = 0; c < 2; c++) {
    let w = Float32Array.from({ length: d }, (_, j) => Math.sin(j * 12.9898 + c * 78.233));
    for (let it = 0; it < 40; it++) {
      const z = new Float32Array(d);
      for (const x of X) { let p = 0; for (let j = 0; j < d; j++) p += x[j] * w[j]; for (let j = 0; j < d; j++) z[j] += x[j] * p; }
      for (const u of comps) { let p = 0; for (let j = 0; j < d; j++) p += z[j] * u[j]; for (let j = 0; j < d; j++) z[j] -= p * u[j]; }
      let nn = 0; for (let j = 0; j < d; j++) nn += z[j] * z[j]; nn = Math.sqrt(nn) || 1;
      for (let j = 0; j < d; j++) w[j] = z[j] / nn;
    }
    comps.push(w);
  }
  const proj = (v) => comps.map((u) => { let p = 0; for (let j = 0; j < v.length; j++) p += (v[j] - mean[j]) * u[j]; return p; });
  return { proj };
}
const map = { pts: [], q: null, hits: [], t0: 0, raf: 0, pca: null };
function layoutMap() {
  const cv = $("#mapCanvas"), r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  cv.width = r.width * dpr; cv.height = r.height * dpr;
  const P = state.vecs.map(map.pca.proj);
  const all = map.q ? P.concat([map.q.raw]) : P;
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const minx = Math.min(...xs), maxx = Math.max(...xs), miny = Math.min(...ys), maxy = Math.max(...ys);
  const pad = 22 * dpr, W = cv.width - pad * 2, H = cv.height - pad * 2;
  map.to = (p) => [pad + ((p[0] - minx) / (maxx - minx || 1)) * W, pad + ((p[1] - miny) / (maxy - miny || 1)) * H];
  map.pts = P.map(map.to);
}
function drawMap(now) {
  const cv = $("#mapCanvas"), g = cv.getContext("2d"), dpr = cv.width / (cv.getBoundingClientRect().width || 1);
  g.clearRect(0, 0, cv.width, cv.height);
  const n = map.pts.length, pages = state.chunks.at(-1).page;
  const t = (now - map.t0) / 1000;
  map.pts.forEach((p, i) => {
    const hit = map.hits.includes(i), appear = Math.min(1, Math.max(0, t * 3 - i * 0.04));
    const hue = 30 + (state.chunks[i].page / pages) * 40;
    g.beginPath(); g.arc(p[0], p[1], (hit ? 6 : 3.5) * dpr * appear, 0, 7);
    g.fillStyle = hit ? "#e0a458" : `hsla(${hue},35%,65%,${0.55 * appear})`; g.fill();
    if (hit) { g.beginPath(); g.arc(p[0], p[1], (10 + 3 * Math.sin(t * 3)) * dpr, 0, 7); g.strokeStyle = "rgba(224,164,88,.45)"; g.lineWidth = dpr; g.stroke(); }
  });
  if (map.q) {
    const qp = map.to(map.q.raw);
    map.hits.forEach((i) => { g.beginPath(); g.moveTo(qp[0], qp[1]); g.lineTo(map.pts[i][0], map.pts[i][1]); g.strokeStyle = "rgba(224,164,88,.35)"; g.lineWidth = dpr; g.setLineDash([4 * dpr, 4 * dpr]); g.stroke(); g.setLineDash([]); });
    g.beginPath(); g.moveTo(qp[0] - 6 * dpr, qp[1]); g.lineTo(qp[0] + 6 * dpr, qp[1]); g.moveTo(qp[0], qp[1] - 6 * dpr); g.lineTo(qp[0], qp[1] + 6 * dpr);
    g.strokeStyle = "#efe7d8"; g.lineWidth = 1.5 * dpr; g.stroke();
  }
  map.raf = requestAnimationFrame(drawMap);
}
function startMap() {
  map.pca = pca2(state.vecs); map.q = null; map.hits = []; map.t0 = performance.now();
  layoutMap(); cancelAnimationFrame(map.raf); map.raf = requestAnimationFrame(drawMap);
  addEventListener("resize", () => state && layoutMap());
}
function pointQuery(r) { map.q = { raw: map.pca.proj(r.qv) }; map.hits = r.found ? r.top.map((t) => t.i) : []; layoutMap(); }

/* ---------- optional: written answers from a small model that runs on the GPU ---------- */
let llm = null;
const HAS_GPU = typeof navigator !== "undefined" && !!navigator.gpu;
async function getLLM() {
  if (llm) return llm;
  const web = await import("https://esm.run/@mlc-ai/web-llm");
  llm = await web.CreateMLCEngine("Qwen2.5-1.5B-Instruct-q4f16_1-MLC", {
    initProgressCallback: (p) => say(`Loading the writing model (one time, about 1 GB) ${Math.round((p.progress || 0) * 100)}%`),
  });
  return llm;
}
async function compose(q, r) {
  const eng = await getLLM();
  const ctx = r.top.map((t) => `[p.${t.c.page}] ${t.c.text}`).join("\n\n");
  const out = await eng.chat.completions.create({
    temperature: 0.1, max_tokens: 320,
    messages: [
      { role: "system", content: "Answer the question using ONLY the passages provided. Cite the page like [p.3] after each claim. If the passages do not contain the answer, reply exactly: NOT_FOUND. Be concise." },
      { role: "user", content: `Passages:\n${ctx}\n\nQuestion: ${q}` },
    ],
  });
  return out.choices[0].message.content.trim();
}

/* ---------- UI ---------- */
function addEmpty() {
  $("#log").innerHTML = `<div class="empty">Ask in your own words. Try a definition, a number, a name or a "why" question. Answers are quoted from your document with the page they came from.</div>`;
}
function addQ(q) { const d = document.createElement("div"); d.className = "q"; d.textContent = q; $("#log").append(d); d.scrollIntoView({ block: "end", behavior: "smooth" }); }
function addA(html, none) { const d = document.createElement("div"); d.className = "a" + (none ? " none" : ""); d.innerHTML = html; $("#log").append(d); d.scrollIntoView({ block: "end", behavior: "smooth" }); return d; }
function highlight(text, q) {
  const ks = new Set(keyWords(q).map((w) => w.replace(/s$/, "")));
  return esc(text).replace(/[A-Za-z0-9][A-Za-z0-9'-]*/g, (w) => (ks.has(w.toLowerCase().replace(/s$/, "")) ? `<mark>${w}</mark>` : w));
}
async function showPage(n) {
  if (!pdfDoc) return;
  $("#pagepane").hidden = false; $("#ppTitle").textContent = `Page ${n} of ${pdfDoc.numPages}`;
  const page = await pdfDoc.getPage(n);
  const vp = page.getViewport({ scale: 1.6 }), cv = $("#ppCanvas");
  cv.width = vp.width; cv.height = vp.height;
  await page.render({ canvasContext: cv.getContext("2d"), viewport: vp }).promise;
}
$("#ppClose").onclick = () => ($("#pagepane").hidden = true);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#pagepane").hidden = true; });
$("#log").addEventListener("click", (e) => { const b = e.target.closest("[data-page]"); if (b) showPage(+b.dataset.page); });

$("#ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#q").value.trim(); if (!q || !state) return;
  $("#q").value = ""; if ($("#log .empty")) $("#log").innerHTML = "";
  addQ(q); $("#go").disabled = true;
  const wait = addA(`<div class="tag">Searching</div>Reading the closest passages`);
  try {
    const r = await answer(q); wait.remove(); pointQuery(r);
    if (!r.found) {
      const near = r.best ? `<div class="more"><details><summary>Closest passage anyway (page ${r.best.c.page}, match ${(r.best.sem * 100).toFixed(0)}%)</summary><div class="p">${esc(r.best.c.text)}</div></details></div>` : "";
      addA(`<div class="tag">Not in this document</div>I could not find an answer to that in the pages you gave me, so I am not going to guess.${near}`, true);
    } else {
      let written = "";
      if ($("#writeOn") && $("#writeOn").checked) {
        try { const t = await compose(q, r); if (t && !/NOT_FOUND/.test(t)) written = `<div class="written"><div class="tag">Written answer (small model, check the quotes below)</div>${esc(t).replace(/\[p\.(\d+)\]/g, '<button class="cite" data-page="$1" type="button">p. $1</button>')}</div>`; say("Ready."); }
        catch (e) { say("The writing model could not start here. Showing quotes only."); }
      }
      const quotes = r.pick.map((x) => `<blockquote class="quote"><p>${highlight(x.s, q)}</p>${pdfDoc ? `<button class="cite" data-page="${x.page}" type="button">p. ${x.page}</button>` : `<span class="cite">section ${x.page}</span>`}</blockquote>`).join("");
      const more = r.top.map((t) => `<div class="p"><small>${pdfDoc ? "page" : "section"} ${t.c.page} &middot; match ${(t.sem * 100).toFixed(0)}%</small>${highlight(t.c.text, q)}</div>`).join("");
      addA(`${written}<div class="tag">Quoted from your document</div>${quotes}<div class="more"><details><summary>Show the ${r.top.length} passages searched</summary>${more}</details></div>`);
    }
  } catch (err) { wait.remove(); addA(`<div class="tag">Error</div>${esc(String(err.message || err))}`, true); }
  $("#go").disabled = false; $("#q").focus();
});

async function load(file) {
  $("#err").hidden = true;
  if (file.size > 60 * 1024 * 1024) return fail("That file is over 60 MB. Try a smaller one.");
  $("#hero").hidden = true; $("#work").hidden = false;
  $("#docName").textContent = file.name; $("#docMeta").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  $("#q").disabled = true; $("#go").disabled = true; addEmpty();
  try {
    step("read", 0);
    const isPdf = /pdf$/i.test(file.name) || file.type === "application/pdf";
    const pages = isPdf ? await readPdf(file) : await readText(file);
    const total = pages.reduce((n, p) => n + p.text.split(/\s+/).filter(Boolean).length, 0);
    if (total < 30) throw new Error("There is almost no selectable text in this file. It may be a scan. Marginalia reads text, not images.");
    step("chunk", 28); say("Splitting into passages");
    const chunks = chunkPages(pages);
    $("#docMeta").textContent = `${pages.length} ${isPdf ? "pages" : "sections"} \u00b7 ${total.toLocaleString()} words \u00b7 ${chunks.length} passages`;
    step("embed", 30);
    await getExtractor();
    const vecs = [], B = 16;
    for (let i = 0; i < chunks.length; i += B) {
      vecs.push(...(await embed(chunks.slice(i, i + B).map((c) => c.text))));
      step("embed", 30 + (Math.min(i + B, chunks.length) / chunks.length) * 68);
      say(`Indexing passage ${Math.min(i + B, chunks.length)} of ${chunks.length}`);
    }
    state = { chunks, vecs }; startMap();
    step("ready", 100); say("Ready. Nothing was uploaded anywhere.");
    $("#q").disabled = false; $("#go").disabled = false; $("#q").focus();
  } catch (err) {
    $("#work").hidden = true; $("#hero").hidden = false; fail(String(err.message || err));
  }
}
const inp = $("#file"), drop = $("#drop");
inp.addEventListener("change", () => inp.files[0] && load(inp.files[0]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && load(e.dataTransfer.files[0]));
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inp.click(); } });
if (HAS_GPU) $("#writeRow").hidden = false;
$("#reset").onclick = () => { cancelAnimationFrame(map.raf); state = null; pdfDoc = null; inp.value = ""; $("#work").hidden = true; $("#hero").hidden = false; };

$("#drop").addEventListener("pointermove", (e) => { const r = e.currentTarget.getBoundingClientRect(); e.currentTarget.style.setProperty("--mx", e.clientX - r.left + "px"); e.currentTarget.style.setProperty("--my", e.clientY - r.top + "px"); });
