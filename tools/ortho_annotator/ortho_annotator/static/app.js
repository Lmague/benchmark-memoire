"use strict";

const state = {
  cfg: null, species: [], colors: {},
  nTiles: 0, nCols: 0, nRows: 0,
  index: 0, tileInfo: null,
  activeSpecies: null,
  showExisting: true,
  mode: "annotate",           // "annotate" | "bbox"
  scale: 1, naturalW: 1, naturalH: 1,
  pending: null,
  raster: null,               // metadata du raster courant
  drag: null,                 // état du glisser pour la bbox
  compareSel: [],             // sélection de la galerie
  coverage: [],               // booléen par tuile (non vide)
  mmPoints: { own: {}, existing: [], thumb_w: 0, thumb_h: 0 },
  existingToken: 0,           // garde anti-tempête pour le calque existant
  existingData: {},           // points existants de la tuile courante (cache)
  lastDir: 1,                 // sens du dernier déplacement (pilote le prefetch)
  // ---- prospection ----
  prospect: { available: false },
  showCands: true,
  cands: [],
  candFilter: null,           // espèce sur laquelle se restreindre, ou null
  candCounts: {},
  mmCands: [],
  candToken: 0,
  refCropM: 1.2,              // emprise au sol des vignettes de référence
  expandedRef: null,          // espèce développée à l'échelle dans le panneau
  loupeTimer: null,
};

const el = (id) => document.getElementById(id);

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text().catch(() => "")}`);
  return res.json();
}
function setSave(t, err) { const s = el("save-status"); s.textContent = t; s.style.color = err ? "var(--danger)" : ""; }

// ---- Démarrage --------------------------------------------------------------

async function boot() {
  state.cfg = await api("/api/config");
  state.species = state.cfg.species;
  state.colors = state.cfg.colors;
  state.raster = state.cfg.raster;
  state.nTiles = state.raster.n_tiles;
  state.nCols = state.raster.n_cols;
  state.nRows = state.raster.n_rows;
  state.prospect = state.cfg.prospect || { available: false };
  state.refCropM = state.cfg.reference_crop_m || 1.2;
  document.body.dataset.panel = state.cfg.panel_position;
  el("jump-input").max = String(state.nTiles - 1);

  buildRasterSelect();
  buildReferencePanel();
  buildLegend();
  buildProspectPanel();
  bindEvents();
  el("btn-existing").textContent = "calque existant : " + (state.showExisting ? "on" : "off");
  loadMinimap();
  await loadCoverage();
  await loadMinimapPoints();
  await refreshZones();
  await loadTile(state.raster.first_nonempty || 0);
  refreshCandCounts();
}

async function loadCoverage() {
  try { const d = await api("/api/coverage"); state.coverage = d.coverage || []; }
  catch (_) { state.coverage = []; }
}

async function loadMinimapPoints(ownOnly) {
  try {
    const d = await api(`/api/minimap-points${ownOnly ? "?own_only=1" : ""}`);
    state.mmPoints.own = d.own || {};
    state.mmPoints.thumb_w = d.thumb_w; state.mmPoints.thumb_h = d.thumb_h;
    if (!d.own_only) state.mmPoints.existing = d.existing || [];
    drawMinimapPoints();
  } catch (_) { /* silencieux */ }
}

function buildRasterSelect() {
  const sel = el("raster-select");
  sel.innerHTML = "";
  state.cfg.available_rasters.forEach((name) => {
    const o = document.createElement("option");
    o.value = name; o.textContent = name;
    if (name === state.cfg.current_raster) o.selected = true;
    sel.appendChild(o);
  });
}

// ---- Panneau de référence ---------------------------------------------------

function buildReferencePanel() {
  const panel = el("reference-cards");
  panel.innerHTML = "";
  const refs = state.cfg.references || {};
  state.species.forEach((code, i) => {
    const ref = refs[code] || { available: false, aerial_count: 0, ground_count: 0 };
    const card = document.createElement("div");
    card.className = "ref-card";
    card.dataset.code = code;
    card.title = `Galerie de ${ref.scientific_name || code} — clic pour agrandir`;
    card.innerHTML = `<div class="ref-head"><span class="swatch" style="background:${state.colors[code]}"></span><span class="ref-code">${code}</span></div>`;
    // Vignette = vue du dessus (dataset) en priorité, sinon photo au sol.
    const kind = (ref.aerial_count || 0) > 0 ? "aerial" : "ground";
    const strip = document.createElement("div");
    strip.className = "ref-strip";
    const n = Math.min(3, (ref.aerial_count || 0) || (ref.ground_count || 0));
    if (n > 0) {
      for (let k = 0; k < n; k++) {
        const img = document.createElement("img");
        img.loading = "lazy";
        img.src = `/api/reference-image?code=${encodeURIComponent(code)}&kind=${kind}&idx=${k}`;
        strip.appendChild(img);
      }
    } else {
      const ph = document.createElement("div");
      ph.className = "placeholder"; ph.textContent = "aucune image";
      strip.appendChild(ph);
    }
    card.appendChild(strip);
    const hint = document.createElement("div");
    hint.className = "keyhint";
    hint.textContent = kind === "aerial" ? `${i + 1} · vue du dessus` : `${i + 1} · au sol`;
    card.appendChild(hint);
    card.addEventListener("click", () => openModal(code, ref));
    card.addEventListener("mouseenter", () => expandReference(code));
    panel.appendChild(card);
  });
  applyReferenceScale();
}

// Une seule espèce est développée à la fois : à la vraie échelle une vignette de
// 1,2 m fait près de 200 px, et six espèces développées feraient un panneau de
// 1300 px de haut. L'espèce développée est celle sur laquelle on travaille —
// espèce active, candidat survolé, ou carte survolée.
function expandReference(code) {
  if (state.expandedRef === code) return;
  state.expandedRef = code;
  document.querySelectorAll("#reference-cards .ref-card").forEach((c) =>
    c.classList.toggle("expanded", c.dataset.code === code));
  applyReferenceScale();
}

// Les vignettes développées sont affichées à la MÊME échelle que la tuile : une
// forme de 30 cm y occupe le même nombre de pixels des deux côtés, sinon la
// comparaison visuelle induit en erreur.
function applyReferenceScale() {
  if (!state.tileInfo) return;
  const b = state.tileInfo.bounds;
  const widthM = b.right - b.left;
  if (!(widthM > 0)) return;
  const pxPerM = (state.naturalW * state.scale) / widthM;
  const panelW = el("reference-panel").clientWidth || 220;
  const size = Math.max(56, Math.min(panelW - 34, state.refCropM * pxPerM));
  document.querySelectorAll("#reference-cards .ref-card.expanded .ref-strip img")
    .forEach((img) => { img.style.width = `${size}px`; img.style.height = `${size}px`; });
  document.querySelectorAll("#reference-cards .ref-card:not(.expanded) .ref-strip img")
    .forEach((img) => { img.style.width = ""; img.style.height = ""; });
  const lo = el("loupe-img");
  lo.style.width = `${size}px`; lo.style.height = `${size}px`;
  el("loupe-note").textContent = `${state.refCropM.toFixed(2)} m de côté — échelle de la vue`;
}

// ---- Légende ----------------------------------------------------------------

function buildLegend() {
  const legend = el("legend");
  legend.innerHTML = "";
  const counts = state.cfg.counts || {};
  state.species.forEach((code, i) => {
    const item = document.createElement("div");
    item.className = "legend-item";
    item.dataset.code = code;
    item.innerHTML = `<span class="swatch" style="background:${state.colors[code]}"></span>` +
      `<span>${i + 1}. ${code}</span><span class="count" data-count="${code}">${counts[code] ?? 0}</span>`;
    item.addEventListener("click", () => toggleActiveSpecies(code));
    legend.appendChild(item);
  });
  refreshActiveUI();
}
function updateCounts(counts) {
  if (!counts) return;
  state.cfg.counts = counts;
  state.species.forEach((code) => {
    const c = document.querySelector(`[data-count="${code}"]`);
    if (c) c.textContent = counts[code] ?? 0;
  });
}

// ---- Espèce active ----------------------------------------------------------

function toggleActiveSpecies(code) {
  state.activeSpecies = state.activeSpecies === code ? null : code;
  refreshActiveUI();
}
function setActiveSpecies(code) { state.activeSpecies = code; refreshActiveUI(); }
function clearActiveSpecies() { state.activeSpecies = null; refreshActiveUI(); }
function refreshActiveUI() {
  const p = el("active-species");
  p.textContent = state.activeSpecies || "aucune espèce";
  p.classList.toggle("active", !!state.activeSpecies);
  if (state.activeSpecies) p.style.background = state.colors[state.activeSpecies];
  else p.style.background = "";
  document.querySelectorAll(".legend-item").forEach((it) =>
    it.classList.toggle("active", it.dataset.code === state.activeSpecies));
  if (state.activeSpecies) expandReference(state.activeSpecies);
}

// ---- Mode -------------------------------------------------------------------

function setMode(mode) {
  state.mode = mode;
  document.body.dataset.mode = mode;
  document.querySelectorAll("#mode-toggle button").forEach((b) =>
    b.classList.toggle("on", b.dataset.mode === mode));
}

// ---- Tuile ------------------------------------------------------------------

async function loadTile(index) {
  index = Math.max(0, Math.min(state.nTiles - 1, index));
  state.lastDir = index >= state.index ? 1 : -1;
  state.index = index;
  state.existingData = {};        // évite d'afficher les points de la tuile précédente
  state.cands = [];
  hideContextMenu();
  const img = el("tile");
  img.src = `/api/tile/${index}.img?dir=${state.lastDir}&r=${encodeURIComponent(state.cfg.current_raster)}`;
  await img.decode().catch(() => {});
  state.naturalW = img.naturalWidth; state.naturalH = img.naturalHeight;
  state.tileInfo = await api(`/api/tile/${index}/info`);
  if (state.scale === 1 && index === (state.raster.first_nonempty || 0)) fitScale();
  else applyScale();
  updateIndicator();
  el("empty-note").classList.toggle("hidden", !state.tileInfo.is_empty);
  renderOverlay();
  updateMinimapRect();
  applyReferenceScale();
  if (state.showExisting) loadExisting();
  if (state.prospect.available && state.showCands) loadCandidates();
}

function updateIndicator() {
  el("tile-indicator").textContent = `${state.index} / ${state.nTiles - 1}`;
  const r = state.tileInfo.row, c = state.tileInfo.col;
  el("grid-pos").textContent = `L${r}/${state.nRows - 1} · C${c}/${state.nCols - 1}`;
  el("jump-input").value = String(state.index);
}

// ---- Zoom -------------------------------------------------------------------

function applyScale() {
  const img = el("tile");
  img.style.width = `${state.naturalW * state.scale}px`;
  img.style.height = `${state.naturalH * state.scale}px`;
  img.style.imageRendering = state.scale > 1.3 ? "pixelated" : "auto";
  renderOverlay();
  applyReferenceScale();
}
function fitScale() {
  const v = el("viewer"), pad = 24;
  state.scale = Math.max(0.02, Math.min((v.clientWidth - pad) / state.naturalW,
                                        (v.clientHeight - pad) / state.naturalH));
  applyScale();
}
function zoom(f) { state.scale = Math.max(0.02, Math.min(12, state.scale * f)); applyScale(); }

// ---- Overlay (points + boxes + candidats) ----------------------------------

function renderOverlay() {
  const ov = el("overlay");
  ov.innerHTML = "";
  if (!state.tileInfo) return;
  const pts = state.tileInfo.points || {};
  Object.keys(pts).forEach((code) =>
    pts[code].forEach((p) => addMarker(p.px, p.py, code, p.fid, false)));
  (state.tileInfo.review_boxes || []).forEach((b) => addBox(b));
  if (state.showExisting) {
    Object.keys(state.existingData).forEach((code) =>
      state.existingData[code].forEach((p) => addMarker(p.px, p.py, code, null, true)));
  }
  if (state.showCands) state.cands.forEach(addCandidate);
}
function pct(v, total) { return `${(v / total) * 100}%`; }

function addMarker(px, py, code, fid, existing) {
  const m = document.createElement("div");
  m.className = "marker" + (existing ? " existing" : "");
  m.style.left = pct(px, state.naturalW); m.style.top = pct(py, state.naturalH);
  const color = state.colors[code] || "#fff";
  if (existing) { m.style.borderColor = color; }
  else {
    m.style.background = color;
    m.dataset.fid = fid; m.dataset.species = code;
    m.title = `${code} #${fid} — clic droit : supprimer`;
    m.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); deletePoint(code, fid); });
  }
  el("overlay").appendChild(m);
}

function addCandidate(c) {
  if (state.candFilter && c.species !== state.candFilter) return;
  const d = document.createElement("div");
  d.className = "cand";
  d.style.left = pct(c.px, state.naturalW); d.style.top = pct(c.py, state.naturalH);
  d.style.borderColor = state.colors[c.species] || "#fff";
  const disagree = c.best_code && c.best_code !== c.species;
  if (disagree) d.classList.add("doubt");
  d.title = `Candidat ${c.species} — score ${c.score}` +
    (disagree ? ` (l'encodeur dirait plutôt ${c.best_code})` : "") +
    `\nclic : accepter comme ${c.species}` +
    "\nMaj+clic : accepter avec l'espèce active" +
    "\nclic droit : écarter";
  // Par défaut on enregistre l'espèce proposée par le détecteur. Utiliser
  // l'espèce active à la place demande un Maj+clic explicite : sans cela, une
  // espèce restée sélectionnée réétiqueterait silencieusement les candidats.
  d.addEventListener("click", (e) => {
    e.stopPropagation();
    acceptCandidate(c, e.shiftKey ? state.activeSpecies : null);
  });
  d.addEventListener("contextmenu", (e) => {
    e.preventDefault(); e.stopPropagation(); rejectCandidate(c);
  });
  d.addEventListener("mouseenter", () => { loupeAtUtm(c.x, c.y); expandReference(c.species); });
  const tag = document.createElement("span");
  tag.className = "cand-tag";
  tag.textContent = c.species;
  tag.style.background = state.colors[c.species] || "#fff";
  d.appendChild(tag);
  el("overlay").appendChild(d);
}

function addBox(b) {
  const d = document.createElement("div");
  d.className = "rbox";
  const x = Math.min(b.px0, b.px1), y = Math.min(b.py0, b.py1);
  const w = Math.abs(b.px1 - b.px0), h = Math.abs(b.py1 - b.py0);
  d.style.left = pct(x, state.naturalW); d.style.top = pct(y, state.naturalH);
  d.style.width = pct(w, state.naturalW); d.style.height = pct(h, state.naturalH);
  const del = document.createElement("button");
  del.className = "rbox-del"; del.textContent = "✕"; del.title = "Supprimer la zone";
  del.addEventListener("click", (e) => { e.stopPropagation(); deleteReview(b.fid); });
  d.appendChild(del);
  el("overlay").appendChild(d);
}

// ---- Calque existant --------------------------------------------------------

async function loadExisting() {
  const idx = state.index;
  const token = ++state.existingToken;           // invalide les réponses obsolètes
  let data;
  try { data = await api(`/api/tile/${idx}/existing`); }
  catch (_) { return; }
  if (token !== state.existingToken || idx !== state.index) return;
  state.existingData = data.available ? (data.points || {}) : {};
  renderOverlay();
}
async function toggleExisting() {
  state.showExisting = !state.showExisting;
  el("btn-existing").textContent = "calque existant : " + (state.showExisting ? "on" : "off");
  if (state.showExisting) await loadExisting();
  else { state.existingData = {}; renderOverlay(); }
}

// ---- Prospection ------------------------------------------------------------

function buildProspectPanel() {
  if (!state.prospect || !state.prospect.available) return;
  el("prospect-bar").classList.remove("hidden");
  el("prospect-section").classList.remove("hidden");
  el("mm-cand-legend").classList.remove("hidden");
  renderProspectList();
  const cal = state.prospect.calibration || {};
  const kinds = [];
  if (state.prospect.has_color) kinds.push("couleur");
  if (state.prospect.has_dense) kinds.push("dense (DINOv3)");
  el("prospect-help").innerHTML =
    `Détecteurs : ${kinds.join(" + ") || "aucun"}.<br>` +
    `R = rappel mesuré, P = précision <b>minorée</b> (les annotations existantes ` +
    `étant incomplètes, un « faux » candidat est parfois une vraie plante).<br>` +
    `<b>S</b> saute au candidat suivant · <b>A</b> analyse la tuile · <b>C</b> affiche/masque.`;
}

function renderProspectList() {
  const list = el("prospect-list");
  list.innerHTML = "";
  const reco = state.prospect.recommendation || {};
  const cal = state.prospect.calibration || { color: {}, dense: {} };
  let total = 0;
  state.species.forEach((code) => {
    const kind = reco[code];
    const c = kind === "dense" ? (cal.dense || {})[code] : (cal.color || {})[code];
    const counts = state.candCounts[code] || {};
    const nNew = counts["new"] || 0;
    total += nNew;
    const row = document.createElement("div");
    row.className = "prospect-item" + (state.candFilter === code ? " active" : "");
    row.title = kind === "aucun"
      ? "Aucun détecteur fiable pour cette espèce d'après la calibration"
      : "Clic : ne montrer que les candidats de cette espèce";
    const quality = c
      ? `R ${c.recall.toFixed(2)} · P≥ ${c.precision.toFixed(2)}`
      : "non calibré";
    row.innerHTML =
      `<span class="swatch" style="background:${state.colors[code]}"></span>` +
      `<span class="pcode">${code}</span>` +
      `<span class="pkind ${kind || ""}">${kind || "—"}</span>` +
      `<span class="pquality muted">${quality}</span>` +
      `<span class="count">${nNew}</span>`;
    row.addEventListener("click", () => {
      state.candFilter = state.candFilter === code ? null : code;
      renderProspectList();
      renderOverlay();
    });
    list.appendChild(row);
  });
  el("prospect-count").textContent = total ? `(${total})` : "";
}

async function refreshCandCounts() {
  if (!state.prospect.available) return;
  try {
    const d = await api("/api/candidates/counts");
    state.candCounts = d.counts || {};
    renderProspectList();
  } catch (_) { /* silencieux */ }
  loadMinimapCandidates();
}

async function loadCandidates() {
  const idx = state.index;
  const token = ++state.candToken;
  let d;
  try { d = await api(`/api/tile/${idx}/candidates`); }
  catch (_) { return; }
  if (token !== state.candToken || idx !== state.index) return;
  state.cands = d.candidates || [];
  renderOverlay();
}

function toggleCands() {
  state.showCands = !state.showCands;
  el("btn-cands").textContent = "candidats : " + (state.showCands ? "on" : "off");
  if (state.showCands) loadCandidates();
  else renderOverlay();
}

async function acceptCandidate(c, overrideSpecies) {
  try {
    setSave("acceptation…");
    const res = await api("/api/candidates/accept", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: c.id, species: overrideSpecies || c.species }) });
    state.cands = state.cands.filter((k) => k.id !== c.id);
    state.tileInfo = await api(`/api/tile/${state.index}/info`);
    renderOverlay();
    updateCounts(res.counts);
    setSave(`${res.species} ajouté ✓`);
    refreshCandCounts();
    loadMinimapPoints(true);
  } catch (err) { setSave("erreur", true); console.error(err); }
}

async function rejectCandidate(c) {
  try {
    await api("/api/candidates/reject", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: c.id }) });
    state.cands = state.cands.filter((k) => k.id !== c.id);
    renderOverlay();
    setSave("candidat écarté");
    refreshCandCounts();
  } catch (err) { setSave("erreur", true); console.error(err); }
}

async function nextCandidate() {
  if (!state.prospect.available) return;
  try {
    setSave("recherche…");
    const q = state.candFilter ? `&species_code=${encodeURIComponent(state.candFilter)}` : "";
    const d = await api(`/api/candidates/next?index=${state.index}${q}`);
    if (d.index === null || d.index === undefined) {
      setSave("aucun candidat — lancez « prospect scan »");
      return;
    }
    setSave(`candidat ${d.species} (score ${Number(d.score).toFixed(2)}) — ${d.remaining} tuile(s)`);
    await loadTile(d.index);
  } catch (err) { setSave("erreur", true); console.error(err); }
}

async function analyzeTile() {
  if (!state.prospect.available) return;
  try {
    setSave("analyse de la tuile…");
    const body = { index: state.index, mode: "auto" };
    if (state.candFilter) body.species = [state.candFilter];
    const d = await api("/api/analyze-tile", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    state.cands = d.candidates || [];
    state.showCands = true;
    el("btn-cands").textContent = "candidats : on";
    renderOverlay();
    setSave(`${d.count} candidat(s) sur cette tuile`);
    refreshCandCounts();
  } catch (err) { setSave("analyse impossible", true); console.error(err); }
}

async function loadMinimapCandidates() {
  if (!state.prospect.available) return;
  try {
    const d = await api("/api/minimap-candidates");
    state.mmCands = d.points || [];
    drawMinimapPoints();
  } catch (_) { /* silencieux */ }
}

// ---- Loupe (vignette à l'échelle des références) ----------------------------

function loupeAtUtm(x, y) {
  el("loupe-img").src =
    `/api/crop?x=${x}&y=${y}&size_m=${state.refCropM}&out=256`;
}
function loupeFromPixel(px, py) {
  if (!state.tileInfo) return;
  const b = state.tileInfo.bounds;
  const x = b.left + (px / state.naturalW) * (b.right - b.left);
  const y = b.top - (py / state.naturalH) * (b.top - b.bottom);
  loupeAtUtm(x, y);
}

// ---- Clics image : points ou bbox ------------------------------------------

function imgToPixel(e) {
  const r = el("tile").getBoundingClientRect();
  return { px: ((e.clientX - r.left) / r.width) * state.naturalW,
           py: ((e.clientY - r.top) / r.height) * state.naturalH };
}

async function onImageClick(e) {
  if (state.mode !== "annotate") return;
  hideContextMenu();
  const { px, py } = imgToPixel(e);
  if (px < 0 || py < 0 || px > state.naturalW || py > state.naturalH) return;
  if (state.activeSpecies) await placePoint(state.activeSpecies, px, py);
  else { state.pending = { px, py }; showContextMenu(e.clientX, e.clientY); }
}

async function placePoint(code, px, py) {
  try {
    setSave("enregistrement…");
    const res = await api("/api/annotate", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index: state.index, px, py, species: code }) });
    addMarker(res.px, res.py, code, res.fid, false);
    updateCounts(res.counts); setSave("enregistré ✓");
    loadMinimapPoints(true);
  } catch (err) { setSave("erreur", true); console.error(err); }
}
async function deletePoint(code, fid) {
  try {
    setSave("suppression…");
    const res = await api("/api/delete", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ species: code, fid }) });
    const m = document.querySelector(`.marker[data-fid="${fid}"][data-species="${code}"]`);
    if (m) m.remove();
    updateCounts(res.counts); setSave("supprimé ✓");
    loadMinimapPoints(true);
  } catch (err) { setSave("erreur", true); console.error(err); }
}
async function undo() {
  try {
    setSave("annulation…");
    const res = await api("/api/undo", { method: "POST" });
    updateCounts(res.counts);
    state.tileInfo = await api(`/api/tile/${state.index}/info`);
    renderOverlay();
    if (state.showExisting) await loadExisting();
    await refreshZones();
    loadMinimapPoints(true);
    setSave(res.undone ? "annulé ✓" : "rien à annuler");
  } catch (err) { setSave("erreur", true); console.error(err); }
}

// ---- Zone à revoir (glisser en mode bbox) ----------------------------------

function onDown(e) {
  if (state.mode !== "bbox" || e.button !== 0) return;
  const p = imgToPixel(e);
  state.drag = { x0: p.px, y0: p.py, x1: p.px, y1: p.py };
  const rub = el("rubber"); rub.classList.remove("hidden");
  updateRubber();
  e.preventDefault();
}
function onMove(e) {
  if (!state.drag) return;
  const p = imgToPixel(e);
  state.drag.x1 = p.px; state.drag.y1 = p.py;
  updateRubber();
}
function updateRubber() {
  const d = state.drag, rub = el("rubber");
  const x = Math.min(d.x0, d.x1), y = Math.min(d.y0, d.y1);
  const w = Math.abs(d.x1 - d.x0), h = Math.abs(d.y1 - d.y0);
  rub.style.left = pct(x, state.naturalW); rub.style.top = pct(y, state.naturalH);
  rub.style.width = pct(w, state.naturalW); rub.style.height = pct(h, state.naturalH);
}
async function onUp() {
  if (!state.drag) return;
  const d = state.drag; state.drag = null;
  el("rubber").classList.add("hidden");
  if (Math.abs(d.x1 - d.x0) < 4 || Math.abs(d.y1 - d.y0) < 4) return;
  try {
    setSave("zone enregistrée…");
    await api("/api/review", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index: state.index, px0: d.x0, py0: d.y0, px1: d.x1, py1: d.y1, note: "" }) });
    state.tileInfo = await api(`/api/tile/${state.index}/info`);
    renderOverlay(); await refreshZones(); setSave("zone ✓");
  } catch (err) { setSave("erreur", true); console.error(err); }
}
async function deleteReview(fid) {
  try {
    await api("/api/review/delete", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ fid }) });
    state.tileInfo = await api(`/api/tile/${state.index}/info`);
    renderOverlay(); await refreshZones(); setSave("zone supprimée ✓");
  } catch (err) { setSave("erreur", true); console.error(err); }
}
async function refreshZones() {
  const data = await api("/api/review/list");
  const list = el("zones-list"); list.innerHTML = "";
  el("zones-count").textContent = data.boxes.length ? `(${data.boxes.length})` : "";
  data.boxes.forEach((b, i) => {
    const row = document.createElement("div");
    row.className = "zone-item";
    const go = document.createElement("button");
    go.className = "zgo";
    go.textContent = `Zone ${i + 1} → tuile ${b.tile_index}`;
    go.addEventListener("click", () => loadTile(b.tile_index));
    const del = document.createElement("button");
    del.className = "zdel"; del.textContent = "✕";
    del.addEventListener("click", () => deleteReview(b.fid));
    row.appendChild(go); row.appendChild(del);
    list.appendChild(row);
  });
}

// ---- Menu contextuel --------------------------------------------------------

function showContextMenu(cx, cy) {
  const menu = el("context-menu");
  menu.innerHTML = `<div class="cm-title">Choisir l'espèce</div>`;
  state.species.forEach((code, i) => {
    const b = document.createElement("button");
    b.innerHTML = `<span class="swatch" style="background:${state.colors[code]}"></span><span>${i + 1}. ${code}</span>`;
    b.addEventListener("click", async () => {
      hideContextMenu();
      if (state.pending) { await placePoint(code, state.pending.px, state.pending.py); state.pending = null; }
    });
    menu.appendChild(b);
  });
  menu.classList.remove("hidden");
  const mw = menu.offsetWidth, mh = menu.offsetHeight;
  menu.style.left = `${Math.min(cx, window.innerWidth - mw - 8)}px`;
  menu.style.top = `${Math.min(cy, window.innerHeight - mh - 8)}px`;
}
function hideContextMenu() { el("context-menu").classList.add("hidden"); }

// ---- Minimap ----------------------------------------------------------------

function loadMinimap() {
  const img = el("minimap-img");
  img.onload = () => { updateMinimapRect(); drawMinimapPoints(); };
  img.src = `/api/thumbnail.png?t=${Date.now()}`;
}
function updateMinimapRect() {
  const img = el("minimap-img");
  if (!img.clientWidth || !state.tileInfo) return;
  const r = state.raster;
  const fx = img.clientWidth / r.width, fy = img.clientHeight / r.height;
  const rect = el("minimap-rect");
  rect.style.left = `${state.tileInfo.col * r.stride_x * fx}px`;
  rect.style.top = `${state.tileInfo.row * r.stride_y * fy}px`;
  rect.style.width = `${Math.max(3, r.tile_px_x * fx)}px`;
  rect.style.height = `${Math.max(3, r.tile_px_y * fy)}px`;
}
function drawMinimapPoints() {
  const img = el("minimap-img");
  const cv = el("minimap-canvas");
  if (!img.clientWidth || !state.mmPoints.thumb_w) return;
  cv.width = img.clientWidth; cv.height = img.clientHeight;
  cv.style.width = `${img.clientWidth}px`; cv.style.height = `${img.clientHeight}px`;
  const sx = img.clientWidth / state.mmPoints.thumb_w;
  const sy = img.clientHeight / state.mmPoints.thumb_h;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  // candidats : petits carrés translucides, en dessous de tout le reste
  if (state.showCands) {
    ctx.fillStyle = "rgba(255,214,64,0.75)";
    state.mmCands.forEach(([x, y, code]) => {
      if (state.candFilter && code !== state.candFilter) return;
      ctx.fillRect(x * sx - 1, y * sy - 1, 2.5, 2.5);
    });
  }
  // existants : petits points gris translucides
  ctx.fillStyle = "rgba(255,255,255,0.55)";
  state.mmPoints.existing.forEach(([x, y]) => {
    ctx.beginPath(); ctx.arc(x * sx, y * sy, 1.1, 0, 6.283); ctx.fill();
  });
  // les miens : par couleur d'espèce, plus gros
  Object.keys(state.mmPoints.own).forEach((code) => {
    ctx.fillStyle = state.colors[code] || "#fff";
    state.mmPoints.own[code].forEach(([x, y]) => {
      ctx.beginPath(); ctx.arc(x * sx, y * sy, 2.2, 0, 6.283); ctx.fill();
    });
  });
}
// Tuile la plus proche contenant du contenu (évite d'atterrir sur du blanc).
function snapToContent(idx) {
  if (!state.coverage.length || state.coverage[idx]) return idx;
  for (let d = 1; d < state.nTiles; d++) {
    if (idx - d >= 0 && state.coverage[idx - d]) return idx - d;
    if (idx + d < state.nTiles && state.coverage[idx + d]) return idx + d;
  }
  return idx;
}
function onMinimapClick(e) {
  const img = el("minimap-img");
  const rect = img.getBoundingClientRect();
  const r = state.raster;
  const px = (e.clientX - rect.left) / rect.width * r.width;
  const py = (e.clientY - rect.top) / rect.height * r.height;
  const col = Math.max(0, Math.min(state.nCols - 1, Math.floor(px / r.stride_x)));
  const row = Math.max(0, Math.min(state.nRows - 1, Math.floor(py / r.stride_y)));
  let idx = Math.max(0, Math.min(state.nTiles - 1, row * state.nCols + col));
  idx = snapToContent(idx);
  loadTile(idx);
}

// ---- Galerie modale + comparaison ------------------------------------------

function fillGrid(gridId, code, kind, count) {
  const grid = el(gridId); grid.innerHTML = "";
  if (!count) {
    const ph = document.createElement("div");
    ph.className = "placeholder"; ph.textContent = "aucune image";
    grid.appendChild(ph);
    return;
  }
  for (let i = 0; i < count; i++) {
    const src = `/api/reference-image?code=${encodeURIComponent(code)}&kind=${kind}&idx=${i}`;
    const img = document.createElement("img");
    img.loading = "lazy"; img.src = src;
    img.title = "Clic : ajouter à la comparaison";
    img.addEventListener("click", () => addToCompare(src));
    grid.appendChild(img);
  }
}
function openModal(code, ref) {
  const sci = (ref.scientific_name || "").replace(/_/g, " ");
  const link = sci
    ? ` — <a href="https://www.inaturalist.org/taxa/search?q=${encodeURIComponent(sci)}" target="_blank" rel="noopener">iNaturalist ↗</a>`
    : "";
  el("modal-title").innerHTML = `${code} · <i>${sci}</i>${link}`;
  state.compareSel = [];
  renderCompareTray();
  fillGrid("modal-grid-aerial", code, "aerial", ref.aerial_count || 0);
  fillGrid("modal-grid-ground", code, "ground", ref.ground_count || 0);
  el("modal").classList.remove("hidden");
}
function addToCompare(src) {
  if (state.compareSel.includes(src)) return;
  if (state.compareSel.length >= 4) state.compareSel.shift();
  state.compareSel.push(src);
  renderCompareTray();
}
function renderCompareTray() {
  const tray = el("compare-tray");
  tray.innerHTML = "";
  tray.classList.toggle("empty", state.compareSel.length === 0);
  state.compareSel.forEach((src) => {
    const img = document.createElement("img");
    img.src = src; img.title = "Retirer de la comparaison";
    img.addEventListener("click", () => {
      state.compareSel = state.compareSel.filter((s) => s !== src);
      renderCompareTray();
    });
    tray.appendChild(img);
  });
}
function closeModal() { el("modal").classList.add("hidden"); }

// ---- Changement d'orthomosaïque --------------------------------------------

async function switchRaster(name) {
  try {
    setSave("chargement ortho…");
    const res = await api("/api/switch-raster", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
    state.raster = res.raster;
    state.cfg.current_raster = res.current_raster;
    state.nTiles = res.raster.n_tiles; state.nCols = res.raster.n_cols; state.nRows = res.raster.n_rows;
    el("jump-input").max = String(state.nTiles - 1);
    state.scale = 1;
    state.mmCands = [];
    loadMinimap();
    await loadCoverage();
    await loadMinimapPoints();
    await refreshZones();
    await loadTile(res.raster.first_nonempty || 0);
    refreshCandCounts();
    setSave("ortho chargée ✓");
  } catch (err) { setSave("erreur ortho", true); console.error(err); }
}

// ---- Événements -------------------------------------------------------------

function bindEvents() {
  el("btn-first").addEventListener("click", () => loadTile(0));
  el("btn-last").addEventListener("click", () => loadTile(state.nTiles - 1));
  el("btn-prev").addEventListener("click", () => loadTile(state.index - 1));
  el("btn-next").addEventListener("click", () => loadTile(state.index + 1));
  el("btn-prev-content").addEventListener("click", () => gotoContent(-1));
  el("btn-next-content").addEventListener("click", () => gotoContent(1));
  el("jump-input").addEventListener("change", (e) => {
    const v = parseInt(e.target.value, 10); if (!Number.isNaN(v)) loadTile(v);
  });
  el("raster-select").addEventListener("change", (e) => switchRaster(e.target.value));

  el("btn-existing").addEventListener("click", toggleExisting);
  el("btn-zoom-in").addEventListener("click", () => zoom(1.25));
  el("btn-zoom-out").addEventListener("click", () => zoom(0.8));
  el("btn-zoom-fit").addEventListener("click", fitScale);
  el("btn-next-cand").addEventListener("click", nextCandidate);
  el("btn-analyze").addEventListener("click", analyzeTile);
  el("btn-cands").addEventListener("click", toggleCands);
  document.querySelectorAll("#mode-toggle button").forEach((b) =>
    b.addEventListener("click", () => setMode(b.dataset.mode)));

  const img = el("tile");
  img.addEventListener("click", onImageClick);
  img.addEventListener("contextmenu", (e) => e.preventDefault());
  img.addEventListener("mousedown", onDown);
  img.addEventListener("mousemove", onTileHover);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);

  el("minimap").addEventListener("click", onMinimapClick);
  window.addEventListener("resize", () => {
    updateMinimapRect(); drawMinimapPoints(); applyReferenceScale();
  });

  el("modal-close").addEventListener("click", closeModal);
  el("modal-backdrop").addEventListener("click", closeModal);

  document.addEventListener("click", (e) => {
    const menu = el("context-menu");
    if (!menu.classList.contains("hidden") && !menu.contains(e.target) && e.target !== img)
      hideContextMenu();
  });
  document.addEventListener("keydown", onKeyDown);
}

// La loupe suit le curseur, mais seulement quand il s'arrête : sans cela chaque
// déplacement déclencherait une lecture fenêtrée inutile.
function onTileHover(e) {
  const p = imgToPixel(e);
  if (state.loupeTimer) clearTimeout(state.loupeTimer);
  state.loupeTimer = setTimeout(() => loupeFromPixel(p.px, p.py), 220);
}

async function gotoContent(step) {
  const res = await api(`/api/nav/next-nonempty?index=${state.index}&step=${step}`);
  loadTile(res.index);
}

function onKeyDown(e) {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "Escape") {
    if (!el("modal").classList.contains("hidden")) return closeModal();
    if (!el("context-menu").classList.contains("hidden")) return hideContextMenu();
    return clearActiveSpecies();
  }
  if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) { e.preventDefault(); return undo(); }
  switch (e.key) {
    case "ArrowLeft": e.preventDefault(); return loadTile(state.index - 1);
    case "ArrowRight": e.preventDefault(); return loadTile(state.index + 1);
    case "ArrowUp": e.preventDefault(); return loadTile(state.index - state.nCols);
    case "ArrowDown": e.preventDefault(); return loadTile(state.index + state.nCols);
    case "n": case "N": return gotoContent(1);
    case "p": case "P": return gotoContent(-1);
    case "e": case "E": return toggleExisting();
    case "b": case "B": return setMode(state.mode === "bbox" ? "annotate" : "bbox");
    case "s": case "S": return nextCandidate();
    case "a": case "A": return analyzeTile();
    case "c": case "C": return toggleCands();
    case "+": case "=": return zoom(1.25);
    case "-": case "_": return zoom(0.8);
    case "0": return clearActiveSpecies();
    default:
      if (/^[1-9]$/.test(e.key)) {
        const idx = parseInt(e.key, 10) - 1;
        if (idx < state.species.length) setActiveSpecies(state.species[idx]);
      }
  }
}

boot().catch((err) => {
  document.body.innerHTML = `<pre style="padding:20px;color:#ff6b6b">Erreur de démarrage : ${err.message}</pre>`;
  console.error(err);
});
