// Builds docs/force_directed_layout.pptx from results/ (CSV, JSON, figures).
//   npm i pptxgenjs   (or use a global install);  node scripts/make_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const R = (f) => path.join(ROOT, "results", f);

// ---------- data ----------------------------------------------------------------
function readCsv(p) {
  const [h, ...rows] = fs.readFileSync(p, "utf8").trim().split("\n").map((l) => l.split(","));
  return rows.map((r) => Object.fromEntries(h.map((k, i) => [k, r[i]])));
}
const summary = readCsv(R("benchmark_summary.csv"));
const runs = readCsv(R("benchmark_runs.csv"));
const demo = JSON.parse(fs.readFileSync(R("demo_net_summary.json"), "utf8"));

const SD = "Steepest descent", CG = "Conjugate gradient (FR)", LM = "Levenberg-Marquardt";
const METHODS = [SD, CG, LM];
const FAMS = ["grid", "net", "tree", "clusters"];
const PLANAR_FAMS = ["grid", "net", "tree"];
const row = (fam, m) => summary.find((s) => s.family === fam && s.method === m);
const median = (a) => {
  const s = [...a].sort((x, y) => x - y);
  const k = s.length;
  return k % 2 ? s[(k - 1) / 2] : (s[k / 2 - 1] + s[k / 2]) / 2;
};

// per-run paired ratios SD / CG, and how often SD and CG end in the same minimum
const byRun = {};
runs.forEach((r) => ((byRun[r.family + "|" + r.seed] ||= {})[r.method] = r));
const pairs = Object.values(byRun);
const iterRatio = median(pairs.map((p) => +p[SD].iters / +p[CG].iters));
const timeRatio = median(pairs.map((p) => +p[SD].seconds / +p[CG].seconds));
const sameMin = (100 * pairs.filter((p) => Math.abs(+p[SD].energy - +p[CG].energy) < 1e-3 * +p[CG].energy).length) / pairs.length;
const nRuns = runs.length;
const nStarts = pairs.length / FAMS.length;
const fmt = (x, d = 0) => Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const all = (m, k) => +row("ALL", m)[k];

// ---------- theme ----------------------------------------------------------------
const THEME = {
  name: "Graph Physics",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2937", lt1: "FFFFFF", dk2: "14213D", lt2: "F1F4F9",
    accent1: "0072B2", accent2: "D55E00", accent3: "009E73", accent4: "E69F00",
    accent5: "2B3A55", accent6: "6B7280", hlink: "0072B2", folHlink: "6B7280",
  },
};
const METHOD_HEX = { [SD]: "E69F00", [CG]: "0072B2", [LM]: "009E73" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Untangling Graphs with Optimization";
const C = pres.SchemeColor;

// ---------- layouts --------------------------------------------------------------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: "14213D" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.7, w: 6.1, h: 2.6, fontSize: 44, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.75, w: 6.0, h: 1.2, fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 34, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Force-directed graph layout  |  Optimization mini project", options: { x: 0.6, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6 },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: "14213D" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 34, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
});

// ---------- helpers --------------------------------------------------------------
const card = (s, name, x, y, w, h, fill = C.background2) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { type: "none" }, objectName: name });
const txt = (s, name, text, o) => s.addText(text, { isTextBox: true, margin: 0, objectName: name, color: C.text1, fontSize: 15, valign: "top", ...o });
const dot = (s, name, x, y, d, color, label, labelColor = C.background1) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { type: "none" }, objectName: name });
  if (label) s.addText(label, { isTextBox: true, x, y, w: d, h: d, align: "center", valign: "middle", bold: true, fontSize: 18, color: labelColor, margin: 0, objectName: name + " label" });
};
const imgFit = (file, w) => {
  // read PNG/GIF pixel size from header
  const b = fs.readFileSync(file);
  const png = b.slice(0, 8).toString("hex") === "89504e470d0a1a0a";
  const [pw, ph] = png ? [b.readUInt32BE(16), b.readUInt32BE(20)] : [b.readUInt16LE(6), b.readUInt16LE(8)];
  return { w, h: (w * ph) / pw };
};
const section = (t) => pres.addSection({ title: t });
const slide = (sec, master = "CONTENT") => pres.addSlide({ masterName: master, sectionTitle: sec });
const shadow = () => ({ type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 });

// =============================== 1. TITLE =========================================
section("Intro");
let s = slide("Intro", "TITLE_DARK");
s.addText("Untangling Graphs with Optimization", { placeholder: "title" });
s.addText("Force-directed layout as unconstrained minimization of an energy function", { placeholder: "body" });
txt(s, "Course tag", "Optimization  |  Mini project", { x: 0.7, y: 6.4, w: 6, h: 0.4, fontSize: 16, color: C.background2 });
card(s, "Hero card", 7.15, 1.75, 5.65, 3.55, C.background1);
{
  const f = imgFit(R("demo_net_before_after.png"), 5.35);
  s.addImage({ path: R("demo_net_before_after.png"), x: 7.3, y: 1.9 + (3.25 - f.h) / 2, w: f.w, h: f.h, altText: "A tangled random layout of a 30-node graph next to the same graph after optimization, with no edge crossings" });
}
txt(s, "Hero caption", `${demo.n} nodes, ${demo.m} edges: ${demo.start_crossings} crossings from a random start, 0 after optimization`, { x: 7.15, y: 5.55, w: 5.65, h: 0.7, fontSize: 14, color: C.background2 });
s.addNotes("Introduce the idea: drawing a graph nicely is an optimization problem. Left to right in this picture is what our code does in about a quarter of a second.");

// =============================== 2. PHYSICAL PICTURE ==============================
section("Model");
s = slide("Model");
s.addText("Nodes repel, edges pull", { placeholder: "title" });
const bullets = [
  [C.accent2, "1", "Nodes are charged particles", "Every pair pushes apart, so nodes never pile up on each other."],
  [C.accent1, "2", "Edges are springs", "Each connected pair is pulled toward a rest length of 1."],
  [C.accent5, "3", "Best picture = lowest energy", "Find all coordinates (x, y) that minimize the total energy E."],
];
bullets.forEach(([col, n, head, body], i) => {
  const y = 1.6 + i * 1.65;
  card(s, `Idea card ${n}`, 0.6, y, 6.0, 1.45);
  dot(s, `Idea number ${n}`, 0.85, y + 0.38, 0.7, col, n);
  txt(s, `Idea title ${n}`, head, { x: 1.8, y: y + 0.22, w: 4.6, h: 0.4, fontSize: 20, bold: true, color: C.text2 });
  txt(s, `Idea text ${n}`, body, { x: 1.8, y: y + 0.7, w: 4.6, h: 0.65, fontSize: 15 });
});
card(s, "Diagram panel", 7.0, 1.6, 5.73, 4.75);
// connected pair: springs pull together
dot(s, "Spring node A", 8.0, 2.55, 0.5, C.accent1);
dot(s, "Spring node B", 11.2, 2.55, 0.5, C.accent1);
s.addShape(pres.shapes.LINE, { x: 8.5, y: 2.8, w: 2.7, h: 0, line: { color: C.accent1, width: 3, dashType: "dash" }, objectName: "Spring" });
s.addShape(pres.shapes.LINE, { x: 8.55, y: 2.25, w: 0.9, h: 0, line: { color: C.accent1, width: 3, endArrowType: "triangle" }, objectName: "Pull arrow left" });
s.addShape(pres.shapes.LINE, { x: 10.45, y: 2.25, w: 0.9, h: 0, flipH: true, line: { color: C.accent1, width: 3, endArrowType: "triangle" }, objectName: "Pull arrow right" });
txt(s, "Spring caption", "Springs pull connected nodes together", { x: 7.3, y: 3.3, w: 5.1, h: 0.4, fontSize: 16, bold: true, color: C.accent1, align: "center" });
// unconnected pair: charges push apart
dot(s, "Charge node A", 8.9, 4.6, 0.5, C.accent2);
dot(s, "Charge node B", 10.3, 4.6, 0.5, C.accent2);
s.addShape(pres.shapes.LINE, { x: 7.7, y: 4.85, w: 1.1, h: 0, flipH: true, line: { color: C.accent2, width: 3, endArrowType: "triangle" }, objectName: "Push arrow left" });
s.addShape(pres.shapes.LINE, { x: 10.9, y: 4.85, w: 1.1, h: 0, line: { color: C.accent2, width: 3, endArrowType: "triangle" }, objectName: "Push arrow right" });
txt(s, "Charge caption", "Charges push every pair apart", { x: 7.3, y: 5.5, w: 5.1, h: 0.4, fontSize: 16, bold: true, color: C.accent2, align: "center" });
s.addNotes("Same intuition as Obsidian's graph view or a network diagram tool. The only thing we add is to make it precise: one number, the total energy, and minimize it.");

// =============================== 3. ENERGY ========================================
s = slide("Model");
s.addText("One number to minimize: the total energy", { placeholder: "title" });
txt(s, "Variables", "E(z) with z = (x1, y1, ..., xn, yn): 2n unknowns, no constraints", { x: 0.6, y: 1.45, w: 6.4, h: 0.5, fontSize: 16, bold: true, color: C.text2 });
const terms = [
  [C.accent1, "Springs (edges)", "½ · k · (d − L)²"],
  [C.accent2, "Repulsion (all pairs)", "½ · c · L² · ln(1 + R² / (d² + ε²))"],
  [C.accent5, "Gravity (centering)", "½ · g · |z|²"],
];
terms.forEach(([col, head, f], i) => {
  const y = 2.1 + i * 1.5;
  card(s, `Term card ${i + 1}`, 0.6, y, 6.4, 1.3);
  dot(s, `Term marker ${i + 1}`, 0.85, y + 0.5, 0.3, col);
  txt(s, `Term title ${i + 1}`, head, { x: 1.35, y: y + 0.17, w: 5.4, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  txt(s, `Term formula ${i + 1}`, f, { x: 1.35, y: y + 0.65, w: 5.5, h: 0.45, fontSize: 18, fontFace: "Cambria" });
});
{
  const f = imgFit(R("energy_terms.png"), 5.4);
  s.addImage({ path: R("energy_terms.png"), x: 7.35, y: 1.75, w: f.w, h: f.h, altText: "Spring energy, repulsion energy and their sum as functions of the distance between two nodes" });
  txt(s, "Plot caption", "d = distance between two nodes. A connected pair settles where spring and repulsion balance. Constants: L = 1, k = 1, c = 0.3, R = 8.", { x: 7.35, y: 1.75 + f.h + 0.15, w: 5.4, h: 0.9, fontSize: 13, color: C.accent6 });
}
s.addNotes("Repulsion is a smooth version of the classic Fruchterman-Reingold law (force ~ 1/d). It is written so that E is a sum of squares, which is what makes Levenberg-Marquardt applicable later.");

// =============================== 4. SYLLABUS ======================================
s = slide("Model");
s.addText("Where the syllabus shows up", { placeholder: "title" });
const syl = [
  ["Module 1", C.accent5, "Line search", "Bounding phase plus golden-section search pick the step length along every direction."],
  ["Module 2", C.accent4, "Steepest descent", "Follow the negative gradient. Simple, but zig-zags in narrow valleys."],
  ["Module 2", C.accent1, "Fletcher-Reeves conjugate gradient", "Blend in the previous direction to stop the zig-zag, with no matrix needed."],
  ["Module 2", C.accent3, "Levenberg-Marquardt", "E is a sum of squares, so J^T J approximates the Hessian and λ blends Newton with gradient steps."],
];
syl.forEach(([mod, col, head, body], i) => {
  const x = i % 2 ? 6.78 : 0.6;
  const y = i < 2 ? 1.55 : 3.7;
  card(s, `Syllabus card ${i + 1}`, x, y, 5.95, 1.95);
  dot(s, `Syllabus marker ${i + 1}`, x + 0.3, y + 0.32, 0.26, col);
  txt(s, `Syllabus module ${i + 1}`, mod, { x: x + 0.7, y: y + 0.27, w: 3, h: 0.35, fontSize: 14, bold: true, color: C.accent6 });
  txt(s, `Syllabus title ${i + 1}`, head, { x: x + 0.3, y: y + 0.7, w: 5.4, h: 0.4, fontSize: 20, bold: true, color: C.text2 });
  txt(s, `Syllabus text ${i + 1}`, body, { x: x + 0.3, y: y + 1.12, w: 5.4, h: 0.75, fontSize: 15 });
});
txt(s, "Syllabus extras", "Also used: analytic gradient and Jacobian (checked against finite differences), convergence rate and rate constant, local versus global minima of a non-convex function.", { x: 0.6, y: 5.95, w: 12.1, h: 0.8, fontSize: 16, color: C.text1 });
s.addNotes("Module 3 (dynamic programming) is not used here. Everything is Modules 1 and 2.");

// =============================== 5. THREE METHODS =================================
section("Method");
s = slide("Method");
s.addText("Three optimizers, one energy function", { placeholder: "title" });
const q = (m) => demo.methods[m].rate_q.toFixed(2);
const cols = [
  [SD, "Steepest descent",
    ["Direction", "d = −∇E"],
    ["Step length", "golden-section line search"],
    ["Behaviour", `Zig-zags along narrow valleys, so it is slow (rate q = ${q(SD)}).`]],
  [CG, "Fletcher-Reeves CG",
    ["Direction", "d' = −g' + β · d\nβ = |g'|² / |g|²"],
    ["Step length", "same line search; restart every 2n steps"],
    ["Behaviour", `Remembers the last direction and avoids the zig-zag (q = ${q(CG)}).`]],
  [LM, "Levenberg-Marquardt",
    ["Step", "(J^T J + λ · D) δ = −J^T r\nD = diag(J^T J)"],
    ["Step length", "none: λ adapts to the gain ratio"],
    ["Behaviour", `Gauss-Newton when λ is small, gradient descent when large (q = ${q(LM)}).`]],
];
cols.forEach(([m, head, a, b, c], i) => {
  const x = 0.6 + i * 4.12;
  card(s, `Method card ${i + 1}`, x, 1.55, 3.9, 4.95);
  dot(s, `Method marker ${i + 1}`, x + 0.3, 1.88, 0.32, "FFFFFF");
  s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.88, w: 0.32, h: 0.32, fill: { color: METHOD_HEX[m] }, line: { type: "none" }, objectName: `Method colour ${i + 1}` });
  txt(s, `Method title ${i + 1}`, head, { x: x + 0.75, y: 1.82, w: 3.0, h: 0.45, fontSize: 20, bold: true, color: C.text2, valign: "middle" });
  [a, b, c].forEach(([lab, body], j) => {
    const y = 2.6 + j * 1.3;
    txt(s, `Method ${i + 1} label ${j + 1}`, lab, { x: x + 0.3, y, w: 3.3, h: 0.3, fontSize: 13, bold: true, color: C.accent6 });
    txt(s, `Method ${i + 1} text ${j + 1}`, body, { x: x + 0.3, y: y + 0.32, w: 3.35, h: 0.9, fontSize: j === 0 ? 16 : 15, fontFace: j === 0 ? "Cambria" : undefined });
  });
});
txt(s, "Rate note", "q = geometric-mean ratio of successive gradient norms near the end of the run (smaller is faster).", { x: 0.6, y: 6.6, w: 12.1, h: 0.35, fontSize: 12, color: C.accent6 });
s.addNotes("All three start from the same random layout. SD and CG share the same line search so the comparison isolates the choice of direction.");

// =============================== 6. SNAPSHOTS =====================================
section("Results");
s = slide("Results");
s.addText("Untangling in four snapshots", { placeholder: "title" });
{
  const f = imgFit(R("demo_net_cg_snapshots.png"), 12.1);
  s.addImage({ path: R("demo_net_cg_snapshots.png"), x: 0.6, y: 1.9, w: f.w, h: f.h, altText: "Four snapshots of conjugate gradient untangling a graph: 242 crossings, then 55, then 6, then 0" });
  txt(s, "Snapshot caption", `Conjugate gradient on a ${demo.n}-node planar network. Orange edges cross another edge: ${demo.start_crossings} crossings at the start, 0 after ${demo.methods[CG].iters} iterations (${demo.methods[CG].seconds.toFixed(2)} s).`, { x: 0.6, y: 1.9 + f.h + 0.3, w: 12.1, h: 0.9, fontSize: 16 });
}
s.addNotes("Most of the untangling happens in the first 15 iterations; the rest is polishing edge lengths until the gradient is tiny.");

// =============================== 7. GIF ===========================================
s = slide("Results");
s.addText("Watch conjugate gradient untangle it", { placeholder: "title" });
{
  const f = imgFit(R("demo_net_cg.gif"), 8.2);
  s.addImage({ path: R("demo_net_cg.gif"), x: 0.6, y: 1.65, w: f.w, h: f.h, altText: "Animation of the graph untangling as the energy decreases" });
  txt(s, "GIF caption", "Press F5 (slideshow) to play. Right: total energy E on log-log axes; the red dot is the current iteration.", { x: 0.6, y: 1.65 + f.h + 0.25, w: 8.2, h: 0.7, fontSize: 14, color: C.accent6 });
}
const stats = [[`${demo.start_crossings} to 0`, "edge crossings"], [`${demo.methods[CG].iters}`, "iterations"], [`${demo.methods[CG].seconds.toFixed(2)} s`, "wall-clock time"]];
stats.forEach(([big, lab], i) => {
  const y = 1.65 + i * 1.65;
  card(s, `Stat card ${i + 1}`, 9.15, y, 3.58, 1.45);
  txt(s, `Stat value ${i + 1}`, big, { x: 9.35, y: y + 0.18, w: 3.2, h: 0.75, fontSize: 36, bold: true, fontFace: "Cambria", color: C.accent1, valign: "middle" });
  txt(s, `Stat label ${i + 1}`, lab, { x: 9.35, y: y + 0.95, w: 3.2, h: 0.35, fontSize: 15, color: C.text1 });
});
s.addNotes("Play the GIF in slideshow mode. If the animation does not play in your viewer, the four-snapshot slide before this one shows the same thing.");

// =============================== 8. CONVERGENCE ===================================
s = slide("Results");
s.addText("Same minimum, very different cost", { placeholder: "title" });
{
  const f = imgFit(R("demo_net_convergence.png"), 12.1);
  s.addImage({ path: R("demo_net_convergence.png"), x: 0.6, y: 1.45, w: f.w, h: f.h, altText: "Energy and gradient norm versus iterations and versus time for the three methods" });
  const y0 = 1.45 + f.h + 0.25;
  METHODS.forEach((m, i) => {
    const x = 0.6 + i * 4.12, d = demo.methods[m];
    card(s, `Run tile ${i + 1}`, x, y0, 3.9, 1.3);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y0 + 0.2, w: 0.26, h: 0.26, fill: { color: METHOD_HEX[m] }, line: { type: "none" }, objectName: `Run tile colour ${i + 1}` });
    txt(s, `Run tile name ${i + 1}`, m, { x: x + 0.65, y: y0 + 0.15, w: 3.1, h: 0.35, fontSize: 15, bold: true, color: C.text2 });
    txt(s, `Run tile value ${i + 1}`, `${fmt(d.iters)} iterations`, { x: x + 0.25, y: y0 + 0.55, w: 3.5, h: 0.4, fontSize: 22, bold: true, fontFace: "Cambria" });
    txt(s, `Run tile detail ${i + 1}`, `${d.seconds.toFixed(2)} s, E = ${d.energy.toFixed(2)}, ${d.crossings} crossings`, { x: x + 0.25, y: y0 + 0.95, w: 3.5, h: 0.3, fontSize: 13, color: C.accent6 });
  });
}
s.addNotes("All three reach E = 112.79, the same layout up to rotation. The difference is only how long it takes. LM has a bump in the gradient norm where it takes a large trust-region step across a saddle.");

// =============================== 9. BENCHMARK =====================================
s = slide("Results");
s.addText(`${nRuns} runs: CG needs about ${Math.round(iterRatio)}x fewer iterations`, { placeholder: "title" });
const chartBase = {
  x: 0, y: 1.5, w: 6.0, h: 4.35, barDir: "col", barGrouping: "clustered", chartColors: METHODS.map((m) => METHOD_HEX[m]),
  showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: "1F2937",
  catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt", catAxisLabelColor: "1F2937",
  valAxisLabelFontSize: 11, valAxisLabelFontFace: "+mn-lt", valAxisLabelColor: "1F2937",
  valGridLine: { color: "D9DEE7", size: 0.75 }, catGridLine: { style: "none" },
  showValue: true, dataLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelColor: "1F2937", dataLabelPosition: "outEnd",
  showTitle: true, titleFontSize: 14, titleFontFace: "+mn-lt", titleColor: "1F2937", barGapWidthPct: 60,
};
const series = (key) => METHODS.map((m) => ({ name: m, labels: FAMS, values: FAMS.map((f) => +row(f, m)[key]) }));
s.addChart(pres.charts.BAR, series("median_iters"), { ...chartBase, x: 0.6, title: "Median iterations to converge (log scale)", valAxisLogScaleBase: 10, dataLabelFormatCode: "#,##0", valAxisLabelFormatCode: "#,##0", valAxisMinVal: 10 });
s.addChart(pres.charts.BAR, series("median_seconds"), { ...chartBase, x: 6.73, title: "Median wall-clock time (seconds)", dataLabelFormatCode: "0.00", valAxisLabelFormatCode: "0.0", valAxisMinVal: 0 });
card(s, "Speedup banner", 0.6, 6.0, 12.13, 0.85, C.background2);
txt(s, "Speedup text", `Steepest descent / CG, median of ${pairs.length} paired starts: ${iterRatio.toFixed(1)}x iterations, ${timeRatio.toFixed(1)}x time.  LM: ${fmt(all(LM, "median_iters"))} iterations (${(all(LM, "median_iters") / all(CG, "median_iters")).toFixed(1)}x CG), similar time to CG.`, { x: 0.85, y: 6.0, w: 11.7, h: 0.85, fontSize: 16, valign: "middle" });
s.addNotes(`Benchmark: 4 graphs x ${nStarts} random starts x 3 methods. Same start for all three methods, same 3000-iteration budget, stop when the gradient norm is below 1e-4. All ${nRuns} runs converged.`);

// =============================== 10. QUALITY ======================================
s = slide("Results");
s.addText("Speed differs, layout quality does not", { placeholder: "title" });
s.addChart(pres.charts.BAR, METHODS.map((m) => ({ name: m, labels: PLANAR_FAMS, values: PLANAR_FAMS.map((f) => +row(f, m).zero_crossings_pct) })), {
  ...chartBase, x: 0.6, y: 1.5, w: 6.4, h: 5.0, title: "Runs ending with zero edge crossings (%), planar graphs",
  valAxisMinVal: 0, valAxisMaxVal: 100, dataLabelFormatCode: '0"%"', valAxisLabelFormatCode: "0",
});
const zr = (m) => all(m, "zero_crossings_pct");
const treeVals = METHODS.map((m) => +row("tree", m).zero_crossings_pct);
const insights = [
  ["Same quality", `Crossing-free layouts on planar graphs: SD ${zr(SD).toFixed(0)}%, CG ${zr(CG).toFixed(0)}%, LM ${zr(LM).toFixed(0)}%. With ${nStarts} starts per graph these gaps are within run-to-run noise.`],
  ["Usually the same minimum", `Steepest descent and CG ended in the same local minimum in ${sameMin.toFixed(0)}% of the ${pairs.length} paired runs.`],
  ["A non-convex energy", `Which minimum you reach depends on the start, not the method: trees fold in ${(100 - Math.max(...treeVals)).toFixed(0)}-${(100 - Math.min(...treeVals)).toFixed(0)}% of starts. Remedy: several random starts, keep the lowest energy.`],
];
insights.forEach(([head, body], i) => {
  const y = 1.55 + i * 1.7;
  txt(s, `Insight title ${i + 1}`, head, { x: 7.45, y, w: 5.3, h: 0.4, fontSize: 20, bold: true, color: C.text2 });
  txt(s, `Insight text ${i + 1}`, body, { x: 7.45, y: y + 0.45, w: 5.3, h: 1.15, fontSize: 15 });
});
s.addNotes("Crossings are only meaningful for planar graphs (a crossing-free drawing exists), so the clustered graph with K6 cliques is excluded from this chart.");

// =============================== 11. LESSONS ======================================
section("Wrap-up");
s = slide("Wrap-up");
s.addText("What went wrong first, and what we learned", { placeholder: "title" });
const lessons = [
  ["Fix the model before the optimizer", "Our first benchmark gave 0% crossing-free layouts on a dense Delaunay mesh. Even starting from the true planar drawing, the optimizer moved to a folded layout with lower energy. So the energy was wrong, not the optimizer. We switched to long-range log repulsion and a sparse planar network, tuning on separate seeds."],
  ["LM is not magic", `Its J^T J step drops second-order residual terms, which is a likely reason it was slower here: spring and repulsion forces balance at a good layout instead of vanishing. Measured: LM converged linearly (q = ${q(LM)}) and needed ${(all(LM, "median_iters") / all(CG, "median_iters")).toFixed(1)}x the iterations of CG (median ${fmt(all(LM, "median_iters"))} vs ${fmt(all(CG, "median_iters"))}).`],
  ["Compare fairly", `All methods get the same start and the same 3,000-iteration budget. LM was first capped at 300 iterations and looked like it failed; it was only slow. We report iterations and wall-clock time, because one CG iteration costs a line search of about 20 energy evaluations and one LM iteration costs a Jacobian.`],
];
lessons.forEach(([head, body], i) => {
  const x = 0.6 + i * 4.12;
  card(s, `Lesson card ${i + 1}`, x, 1.55, 3.9, 4.9);
  dot(s, `Lesson number ${i + 1}`, x + 0.3, 1.85, 0.6, i === 0 ? C.accent2 : i === 1 ? C.accent3 : C.accent1, String(i + 1));
  txt(s, `Lesson title ${i + 1}`, head, { x: x + 0.3, y: 2.65, w: 3.35, h: 0.8, fontSize: 19, bold: true, color: C.text2 });
  txt(s, `Lesson text ${i + 1}`, body, { x: x + 0.3, y: 3.5, w: 3.4, h: 2.8, fontSize: 15 });
});
s.addNotes("Be upfront about these in the viva: the first model did not work, we diagnosed it by starting the optimizer from the known-good drawing, and we fixed the energy rather than tuning the optimizer.");

// =============================== 12. IMPLEMENTATION ===============================
s = slide("Wrap-up");
s.addText("Implementation and reproducibility", { placeholder: "title" });
card(s, "Repo card", 0.6, 1.55, 6.3, 4.4);
const repo = [
  "forcelayout/",
  "  graphs.py      test graphs, random starts",
  "  energy.py      E, gradient, residuals, J",
  "  linesearch.py  bounding + golden section",
  "  optimizers.py  SD, FR-CG, LM",
  "  metrics.py     crossings, edge CV, rate",
  "  viz.py         snapshots, GIF, plots",
  "scripts/",
  "  run_demo.py    figures + animation",
  "  benchmark.py   multi-start comparison",
  "  make_deck.js   this presentation",
  "tests/           automated tests",
];
txt(s, "Repo tree", repo.join("\n"), { x: 0.9, y: 1.8, w: 5.8, h: 3.9, fontSize: 15, fontFace: "Courier New", lineSpacingMultiple: 1.15 });
const impl = [
  ["Analytic gradient and Jacobian", "Verified against finite differences in the tests."],
  ["Optimizers written from scratch", "NumPy only; no scipy.optimize."],
  ["Reproducible", "Fixed seeds. run_demo.py and benchmark.py regenerate every figure and table in this deck."],
];
impl.forEach(([head, body], i) => {
  const y = 1.65 + i * 1.4;
  txt(s, `Impl title ${i + 1}`, head, { x: 7.35, y, w: 5.35, h: 0.4, fontSize: 20, bold: true, color: C.text2 });
  txt(s, `Impl text ${i + 1}`, body, { x: 7.35, y: y + 0.5, w: 5.35, h: 0.95, fontSize: 15 });
});
s.addNotes("Skip this slide if you are short on time; the repository README has the same information.");

// =============================== 13. TAKEAWAYS ====================================
s = slide("Wrap-up", "CLOSING_DARK");
s.addText("Takeaways", { placeholder: "title" });
const take = [
  ["Layout is optimization", "Minimize a spring plus repulsion energy over 2n coordinates, using the Module 1 and 2 toolbox."],
  ["CG is the sweet spot", `About ${Math.round(iterRatio)}x faster than steepest descent, with no Jacobian or matrix solve.`],
  ["The landscape is non-convex", "All methods reach the same quality; use several random starts and keep the lowest energy."],
];
take.forEach(([head, body], i) => {
  const y = 1.6 + i * 1.5;
  txt(s, `Takeaway number ${i + 1}`, String(i + 1), { x: 0.6, y, w: 0.9, h: 1.1, fontSize: 54, bold: true, fontFace: "Cambria", color: C.accent4, valign: "middle" });
  txt(s, `Takeaway title ${i + 1}`, head, { x: 1.7, y, w: 10.9, h: 0.5, fontSize: 24, bold: true, color: C.background1 });
  txt(s, `Takeaway text ${i + 1}`, body, { x: 1.7, y: y + 0.55, w: 10.9, h: 0.6, fontSize: 17, color: C.background2 });
});
txt(s, "Next steps", "Next: Barnes-Hut approximation for O(n log n) repulsion, momentum methods, and constraint-based layouts.", { x: 0.6, y: 6.3, w: 12.1, h: 0.5, fontSize: 16, color: C.background2 });
s.addNotes("Close on the three takeaways, then take questions.");

// ---------- write ---------------------------------------------------------------
(async () => {
  const out = path.join(ROOT, "docs", "force_directed_layout.pptx");
  fs.mkdirSync(path.dirname(out), { recursive: true });
  await pres.writeFile({ fileName: out });
  try {
    const { applyTheme } = require(process.env.APPLY_THEME || "/mnt/skills/public/pptx/scripts/apply_theme.js");
    await applyTheme(out, THEME);
  } catch (e) {
    console.warn("theme colours not applied (apply_theme.js not found):", e.message);
  }
  console.log("wrote", out);
})();
