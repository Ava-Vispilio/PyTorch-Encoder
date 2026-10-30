/* global Chart */

const state = {
  steps: [],
  program: [],
  replayTimer: null,
  charts: {},
};

function log(msg) {
  const el = document.getElementById("eventLog");
  const line = document.createElement("div");
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  el.prepend(line);
}

function setConn(live) {
  const b = document.getElementById("connBadge");
  b.textContent = live ? "live" : "offline";
  b.className = "badge " + (live ? "live" : "off");
}

function renderRegs(regs) {
  const grid = document.getElementById("regGrid");
  grid.innerHTML = "";
  (regs || [0, 0, 0, 0]).forEach((v, i) => {
    const d = document.createElement("div");
    d.className = "reg";
    d.innerHTML = `<span>R${i}</span><strong>${v}</strong>`;
    grid.appendChild(d);
  });
}

function renderProgram(programOps, pc) {
  const list = document.getElementById("instrList");
  list.innerHTML = "";
  (programOps || []).forEach((op, idx) => {
    const li = document.createElement("li");
    const args = [op.a, op.b, op.c].filter((x) => x !== 0).join(", ");
    li.textContent = `${idx}: ${op.op}${args ? " " + args : ""}`;
    if (idx === pc) li.classList.add("active");
    else if (idx < pc) li.classList.add("done");
    list.appendChild(li);
  });
  const bar = document.querySelector("#pcBar i");
  const pct = programOps?.length ? (100 * (pc || 0)) / programOps.length : 0;
  bar.style.width = `${Math.min(pct, 100)}%`;
}

function applyStep(ev) {
  document.getElementById("vStep").textContent = ev.step ?? "—";
  document.getElementById("vOp").textContent = ev.expected_op ?? ev.op ?? "—";
  document.getElementById("vPc").textContent =
    ev.pc_after ?? ev.pc ?? ev.next_pc ?? "—";
  document.getElementById("vHalt").textContent = String(ev.halted ?? false);
  document.getElementById("vAux").textContent =
    ev.aux_loss != null ? Number(ev.aux_loss).toFixed(6) : "—";
  document.getElementById("vTask").textContent =
    ev.task_loss != null ? Number(ev.task_loss).toFixed(4) : "—";
  renderRegs(ev.regs);
  if (ev.out_preview != null) {
    document.getElementById("outPreview").textContent = ev.out_preview;
  }
  const pc = ev.pc_before ?? ev.pc ?? 0;
  renderProgram(state.program, pc);

  state.steps.push(ev);
  updateTimelineChart();
}

function ensureTimelineChart() {
  if (state.charts.timeline) return state.charts.timeline;
  const ctx = document.getElementById("chartTimeline");
  state.charts.timeline = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Task loss",
          data: [],
          borderColor: "#38bdf8",
          tension: 0.2,
          yAxisID: "y",
        },
        {
          label: "Aux loss",
          data: [],
          borderColor: "#6ee7b7",
          tension: 0.2,
          yAxisID: "y2",
        },
        {
          label: "PC",
          data: [],
          borderColor: "#fbbf24",
          stepped: true,
          yAxisID: "y3",
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: { position: "left", title: { display: true, text: "task" } },
        y2: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "aux" } },
        y3: { display: false },
      },
      plugins: { legend: { labels: { color: "#8b9cb3" } } },
    },
  });
  return state.charts.timeline;
}

function updateTimelineChart() {
  const chart = ensureTimelineChart();
  const tail = state.steps.slice(-200);
  chart.data.labels = tail.map((s) => s.step);
  chart.data.datasets[0].data = tail.map((s) => s.task_loss ?? null);
  chart.data.datasets[1].data = tail.map((s) => s.aux_loss ?? null);
  chart.data.datasets[2].data = tail.map((s) => s.pc_before ?? s.pc ?? null);
  chart.update("none");
}

function ensureAccuracyChart() {
  if (state.charts.accuracy) return state.charts.accuracy;
  const ctx = document.getElementById("chartAccuracy");
  state.charts.accuracy = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Baseline test acc %", data: [], borderColor: "#38bdf8", tension: 0.2 },
        { label: "Trojan + VM test acc %", data: [], borderColor: "#f87171", tension: 0.2 },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 80, max: 100, title: { display: true, text: "accuracy %" } },
      },
      plugins: { legend: { labels: { color: "#8b9cb3" } } },
    },
  });
  return state.charts.accuracy;
}

function renderBenchmark(data) {
  if (!data || !data.series) return;
  const chart = ensureAccuracyChart();
  chart.data.labels = data.series.map((p) => p.step);
  chart.data.datasets[0].data = data.series.map((p) => p.baseline_acc);
  chart.data.datasets[1].data = data.series.map((p) => p.trojan_acc);
  chart.update();

  const table = document.getElementById("accTable");
  const last = data.series[data.series.length - 1];
  table.innerHTML = `
    <tr><th>Run</th><th>Final test accuracy</th><th>Steps</th></tr>
    <tr><td>Baseline (no trojan)</td><td>${last.baseline_acc.toFixed(2)}%</td><td>${last.step}</td></tr>
    <tr><td>Trojan + VM aux</td><td>${last.trojan_acc.toFixed(2)}%</td><td>${last.step}</td></tr>
    <tr><td>Δ (trojan − baseline)</td><td colspan="2">${(last.trojan_acc - last.baseline_acc).toFixed(2)} pp</td></tr>
  `;
  log(`Benchmark loaded (${data.series.length} points)`);
}

function connectSSE() {
  const es = new EventSource("/events");
  es.addEventListener("open", () => {
    setConn(true);
    log("SSE connected — waiting for training steps…");
  });
  es.addEventListener("step", (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === "program") {
      state.program = ev.ops || [];
      renderProgram(state.program, 0);
      return;
    }
    if (ev.type === "server") {
      log(`Server: ${ev.url}`);
      return;
    }
    applyStep(ev);
  });
  es.onerror = () => {
    setConn(false);
    log("SSE disconnected (replay or restart training with --ui)");
  };
}

async function loadTrace() {
  const res = await fetch("/api/trace");
  const data = await res.json();
  state.steps = [];
  const steps = data.steps || [];
  if (!steps.length) {
    log("No trace file yet — run train_vm.py --ui");
    return;
  }
  log(`Replay ${steps.length} steps from ${data.path}`);
  let i = 0;
  clearInterval(state.replayTimer);
  const speed = () => Number(document.getElementById("replaySpeed").value);
  state.replayTimer = setInterval(() => {
    if (i >= steps.length) {
      clearInterval(state.replayTimer);
      return;
    }
    applyStep(steps[i++]);
  }, speed());
}

async function loadBenchmark() {
  const res = await fetch("/api/benchmark");
  const data = await res.json();
  if (data.error) {
    log(data.error + " — run: python benchmark_accuracy.py");
    return;
  }
  renderBenchmark(data);
}

async function loadManifest() {
  const res = await fetch("/api/manifest");
  const data = await res.json();
  if (data.program && data.program.ops) {
    state.program = data.program.ops;
    renderProgram(state.program, 0);
  }
}

document.getElementById("btnReplay").addEventListener("click", loadTrace);
document.getElementById("btnBenchmark").addEventListener("click", loadBenchmark);

loadManifest();
loadBenchmark();
connectSSE();
