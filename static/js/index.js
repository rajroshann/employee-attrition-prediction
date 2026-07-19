/* =============================================
   index.js — Modules 3–8 (Prediction page)
   ============================================= */

/* ── Global state — one source of truth for PDF ── */
let STATE = {
  formData  : null,
  prediction: null,
  shapResult: null,
  recs      : null,
  cost      : null,
};

/* ── Chart instances (destroyed on re-predict) ── */
let shapChartInstance = null;
let fiChartInstance   = null;


/* ════════════════════════════════════
   Helpers
════════════════════════════════════ */

/* Format number as Indian locale */
function fmt(n) {
  return Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 });
}

/* Reveal a result card with staggered animation */
function showCard(id, delay = 0) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.add('visible');
  setTimeout(() => el.classList.add('shown'), delay + 30);
}

/* Collect all 11 form values into a plain object */
function collectForm() {
  return {
    age                     : document.getElementById('f_age').value,
    gender                  : document.getElementById('f_gender').value,
    department              : document.getElementById('f_department').value,
    job_role                : document.getElementById('f_job_role').value,
    monthly_income          : document.getElementById('f_income').value,
    overtime                : document.getElementById('f_overtime').value,
    job_satisfaction        : document.getElementById('f_job_sat').value,
    environment_satisfaction: document.getElementById('f_env_sat').value,
    work_life_balance       : document.getElementById('f_wlb').value,
    years_at_company        : document.getElementById('f_years').value,
    distance_from_home      : document.getElementById('f_distance').value,
  };
}


/* ════════════════════════════════════
   Module 3 — Run full prediction pipeline
════════════════════════════════════ */
async function runPrediction() {
  const btn     = document.getElementById('predictBtn');
  const spinner = document.getElementById('spinner');
  const btnText = document.getElementById('btnText');

  /* Loading state */
  btn.disabled         = true;
  spinner.style.display = 'block';
  btnText.textContent  = 'Predicting…';

  STATE.formData = collectForm();

  try {
    /* ── A: Predict ── */
    const predRes  = await fetch('/api/predict', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify(STATE.formData),
    });
    STATE.prediction = await predRes.json();
    renderPrediction(STATE.prediction);
    showCard('resultCard', 0);

    /* ── B: SHAP Explanation ── */
    const shapRes  = await fetch('/api/explain', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify(STATE.formData),
    });
    const shapJson = await shapRes.json();
    STATE.shapResult = shapJson.result;
    renderSHAP(STATE.shapResult);
    showCard('shapCard', 100);

    /* ── C: Recommendations ── */
    const recRes  = await fetch('/api/recommend', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ shap_values: STATE.shapResult.shap_values }),
    });
    const recJson = await recRes.json();
    STATE.recs = recJson.recommendations;
    renderRecommendations(STATE.recs);
    showCard('recCard', 200);

    /* ── D: Cost Calculator ── */
    const costRes  = await fetch('/api/cost-calculator', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({
        monthly_income: STATE.formData.monthly_income,
        job_level     : 2,
      }),
    });
    STATE.cost = await costRes.json();
    renderCost(STATE.cost);
    showCard('costCard', 300);

    /* ── E: Reveal download button ── */
    showCard('dlCard', 400);

  } catch (err) {
    console.error('Prediction error:', err);
    alert('Something went wrong. Make sure app.py is running.');
  } finally {
    btn.disabled          = false;
    spinner.style.display = 'none';
    btnText.textContent   = '🔮 Predict Again';
  }
}


/* ════════════════════════════════════
   Render: Prediction result (Module 3 output)
════════════════════════════════════ */
function renderPrediction(p) {
  const isLeave = p.label === 'Leave';

  document.getElementById('predLabel').textContent = p.label;
  document.getElementById('predLabel').className   = 'pred-label ' + (isLeave ? 'pred-leave' : 'pred-stay');
  document.getElementById('probPct').textContent   = p.probability + '%';

  const bar       = document.getElementById('probBar');
  bar.style.width = p.probability + '%';
  bar.className   = 'prob-bar' + (isLeave ? '' : ' stay');

  const badge       = document.getElementById('riskBadge');
  badge.textContent = p.risk_level + ' Risk';
  badge.className   = 'risk-badge risk-' + p.risk_level.toLowerCase();
}


/* ════════════════════════════════════
   Render: SHAP chart (Module 4)
════════════════════════════════════ */
function renderSHAP(result) {
  if (shapChartInstance) { shapChartInstance.destroy(); }

  const vals   = result.shap_values;
  const labels = vals.map(v => v.label);
  const data   = vals.map(v => v.shap_value);
  const colors = data.map(v =>
    v > 0 ? 'rgba(244,63,94,.85)' : 'rgba(16,185,129,.85)'
  );

  shapChartInstance = new Chart(document.getElementById('shapChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors,
        borderRadius   : 6,
        borderSkipped  : false,
      }],
    },
    options: {
      responsive         : true,
      maintainAspectRatio: false,
      indexAxis          : 'y',
      plugins: {
        legend : { display: false },
        tooltip: {
          backgroundColor: '#1A2740',
          borderColor    : '#1F3050',
          borderWidth    : 1,
          titleColor     : '#F1F5F9',
          bodyColor      : '#94A3B8',
          padding        : 10,
          callbacks: {
            label: ctx => {
              const dir = ctx.parsed.x > 0 ? ' ▲ increases risk' : ' ▼ decreases risk';
              return ` ${ctx.parsed.x.toFixed(4)}${dir}`;
            },
          },
        },
      },
      scales: {
        x: { grid: { color: 'rgba(31,48,80,.8)' }, ticks: { color: '#94A3B8' } },
        y: { grid: { color: 'transparent'       }, ticks: { color: '#94A3B8' } },
      },
    },
  });
}


/* ════════════════════════════════════
   Render: Recommendations (Module 6)
════════════════════════════════════ */
function renderRecommendations(recs) {
  const list = document.getElementById('recList');
  if (!recs || recs.length === 0) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="icon">✅</div>
        <p>No critical risk factors found.</p>
      </div>`;
    return;
  }
  list.innerHTML = recs.map(r => `
    <div class="rec-item">
      <div class="rec-feature">${r.feature}</div>
      <div class="rec-title">${r.title}</div>
      <div class="rec-detail">${r.detail}</div>
    </div>
  `).join('');
}


/* ════════════════════════════════════
   Render: Cost Calculator (Module 7)
════════════════════════════════════ */
function renderCost(c) {
  document.getElementById('costAnnual').textContent = '₹' + fmt(c.annual_salary);
  document.getElementById('costMult').textContent   = c.multiplier + '×';
  document.getElementById('costTotal').textContent  = '₹' + fmt(c.replacement_cost);
  document.getElementById('costNote').textContent   = c.note;

  const bd    = c.breakdown;
  const total = c.replacement_cost;
  const items = [
    { label: 'Recruitment & Hiring',  key: 'recruitment'   },
    { label: 'Onboarding & Training', key: 'onboarding'    },
    { label: 'Lost Productivity',     key: 'productivity'  },
    { label: 'Overtime Coverage',     key: 'overtime_cover' },
  ];

  document.getElementById('costBreakdown').innerHTML = items.map(i => `
    <div class="cb-row">
      <span class="cb-label">${i.label}</span>
      <span class="cb-val">₹${fmt(bd[i.key])}</span>
    </div>
    <div class="cb-bar-wrap">
      <div class="cb-bar" style="width:${(bd[i.key] / total * 100).toFixed(1)}%"></div>
    </div>
  `).join('');
}


/* ════════════════════════════════════
   Module 5 — Feature Importance (loads on page load)
════════════════════════════════════ */
async function loadFeatureImportance() {
  const res  = await fetch('/api/feature-importance');
  const json = await res.json();
  const data = json.data;

  const labels = data.map(d => d.label);
  const values = data.map(d => d.importance);
  const colors = values.map((_, i) => {
    const alpha = 0.9 - (i / values.length) * 0.45;
    return `rgba(99,102,241,${alpha.toFixed(2)})`;
  });

  fiChartInstance = new Chart(document.getElementById('fiChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data           : values,
        backgroundColor: colors,
        borderRadius   : 5,
        borderSkipped  : false,
      }],
    },
    options: {
      responsive         : true,
      maintainAspectRatio: false,
      indexAxis          : 'y',
      plugins: {
        legend : { display: false },
        tooltip: {
          backgroundColor: '#1A2740',
          borderColor    : '#1F3050',
          borderWidth    : 1,
          titleColor     : '#F1F5F9',
          bodyColor      : '#94A3B8',
          padding        : 10,
          callbacks: {
            label: ctx => ` Importance: ${ctx.parsed.x.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: { grid: { color: 'rgba(31,48,80,.8)' }, ticks: { color: '#94A3B8' } },
        y: { grid: { color: 'transparent'        }, ticks: { color: '#94A3B8' } },
      },
    },
  });
}


/* ════════════════════════════════════
   Module 8 — Download PDF Report
════════════════════════════════════ */
async function downloadReport() {
  if (!STATE.prediction) {
    alert('Please run a prediction first.');
    return;
  }

  const btn         = document.getElementById('dlBtn');
  btn.textContent   = '⏳ Generating PDF…';
  btn.disabled      = true;

  try {
    const res = await fetch('/api/download-report', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({
        form_data      : STATE.formData,
        prediction     : STATE.prediction,
        shap_result    : STATE.shapResult,
        recommendations: STATE.recs,
        cost           : STATE.cost,
      }),
    });

    /* Trigger browser file download */
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'attrition_report.pdf';
    a.click();
    URL.revokeObjectURL(url);

  } catch (err) {
    alert('PDF generation failed: ' + err.message);
  } finally {
    btn.textContent = '⬇️ Download PDF Report';
    btn.disabled    = false;
  }
}


/* ════════════════════════════════════
   Mobile sidebar toggle
════════════════════════════════════ */
function initSidebar() {
  const hamburger = document.getElementById('hamburger');
  const sidebar   = document.getElementById('sidebar');
  const overlay   = document.getElementById('sidebarOverlay');

  if (!hamburger) return;

  hamburger.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    overlay.classList.toggle('active');
  });
  overlay.addEventListener('click', () => {
    sidebar.classList.remove('open');
    overlay.classList.remove('active');
  });
}


/* ════════════════════════════════════
   Bootstrap on DOM ready
════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initSidebar();
  loadFeatureImportance(); /* Module 5 — no form input needed */
});