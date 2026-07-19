/* =============================================
   dashboard.js — Module 1 (KPIs) + Module 2 (Charts)
   ============================================= */

/* ── Chart.js global defaults ── */
Chart.defaults.color         = '#94A3B8';
Chart.defaults.font.family   = 'Inter, system-ui, sans-serif';
Chart.defaults.font.size     = 11;
Chart.defaults.plugins.legend.display = false;

/* ── Shared tooltip config ── */
const TOOLTIP = {
  backgroundColor: '#1A2740',
  borderColor    : '#1F3050',
  borderWidth    : 1,
  titleColor     : '#F1F5F9',
  bodyColor      : '#94A3B8',
  padding        : 10,
};

/* ── Base chart options factory ── */
const baseOpts = () => ({
  responsive         : true,
  maintainAspectRatio: false,
  plugins: {
    legend : { display: false },
    tooltip: TOOLTIP,
  },
});

/* ── Colour palette ── */
const C = {
  indigo : 'rgba(99,102,241,.85)',
  emerald: 'rgba(16,185,129,.85)',
  rose   : 'rgba(244,63,94,.85)',
  amber  : 'rgba(245,158,11,.85)',
  cyan   : 'rgba(6,182,212,.85)',
  purple : 'rgba(168,85,247,.85)',
};
const B = {
  indigo : 'rgb(99,102,241)',
  emerald: 'rgb(16,185,129)',
};

/* ── Shared scale options ── */
const scaleOpts = {
  x: { grid: { color: 'rgba(31,48,80,.8)' }, ticks: { color: '#94A3B8' } },
  y: { grid: { color: 'rgba(31,48,80,.8)' }, ticks: { color: '#94A3B8' } },
};


/* ── Chart helper: bar (vertical or horizontal) ── */
function barChart(id, labels, values, color, horiz = false) {
  new Chart(document.getElementById(id), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data           : values,
        backgroundColor: color,
        borderRadius   : 6,
        borderSkipped  : false,
      }],
    },
    options: {
      ...baseOpts(),
      indexAxis: horiz ? 'y' : 'x',
      scales   : scaleOpts,
    },
  });
}

/* ── Chart helper: donut ── */
function donutChart(id, labels, values, colors) {
  new Chart(document.getElementById(id), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data           : values,
        backgroundColor: colors,
        borderWidth    : 0,
        borderRadius   : 4,
        hoverOffset    : 8,
      }],
    },
    options: {
      ...baseOpts(),
      cutout : '60%',
      plugins: {
        legend: {
          display : true,
          position: 'bottom',
          labels  : { color: '#94A3B8', boxWidth: 12, padding: 16 },
        },
        tooltip: {
          ...TOOLTIP,
          callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed}%` },
        },
      },
    },
  });
}


/* ════════════════════════════════════
   Module 1 — Load KPI stats
════════════════════════════════════ */
async function loadDashboardStats() {
  const res  = await fetch('/api/dashboard-stats');
  const json = await res.json();
  const s    = json.stats;
  const m    = json.model_metrics;

  /* KPI values */
  document.getElementById('kpiAttrPct').textContent   = s.attrition_pct + '%';
  document.getElementById('kpiAttrCount').textContent = s.attrition_count + ' employees left';
  document.getElementById('ringPct').textContent      = s.attrition_pct + '%';
  document.getElementById('kpiTotal').textContent     = s.total_employees.toLocaleString();
  document.getElementById('kpiIncome').textContent    = '₹' + Number(s.avg_monthly_income)
    .toLocaleString('en-IN', { maximumFractionDigits: 0 });
  document.getElementById('kpiExp').textContent       = s.avg_experience + ' yrs';
  document.getElementById('kpiAge').textContent       = s.avg_age;

  /* Gender ratio */
  const male   = s.gender_ratio['Male']   || 0;
  const female = s.gender_ratio['Female'] || 0;
  document.getElementById('kpiGender').textContent    = (male / (male + female) * 100).toFixed(0) + '% M';
  document.getElementById('kpiGenderSub').textContent = `${male} Male / ${female} Female`;

  /* Attrition ring (small donut inside KPI hero) */
  new Chart(document.getElementById('attritionRing'), {
    type: 'doughnut',
    data: {
      datasets: [{
        data           : [s.attrition_pct, 100 - s.attrition_pct],
        backgroundColor: ['rgb(99,102,241)', 'rgba(99,102,241,.12)'],
        borderWidth    : 0,
        borderRadius   : 4,
      }],
    },
    options: {
      responsive         : true,
      maintainAspectRatio: false,
      cutout             : '72%',
      plugins: {
        legend : { display: false },
        tooltip: { enabled: false },
      },
    },
  });

  /* Department progress bars */
  const dept      = s.dept_counts;
  const deptTotal = Object.values(dept).reduce((a, b) => a + b, 0);
  const deptColors = ['var(--indigo)', 'var(--cyan)', 'var(--amber)'];

  document.getElementById('deptList').innerHTML = Object.entries(dept)
    .map(([name, count], i) => `
      <div class="dept-item">
        <span class="dept-name">${name}</span>
        <span class="dept-count">${count}</span>
      </div>
      <div class="dept-bar-wrap">
        <div class="dept-bar" style="width:${(count / deptTotal * 100).toFixed(1)}%; background:${deptColors[i]}"></div>
      </div>
    `).join('');

  /* Model metrics strip */
  document.getElementById('mRecall').textContent = m.recall;
  document.getElementById('mF1').textContent     = m.f1_score;
  document.getElementById('mAuc').textContent    = m.roc_auc;
  document.getElementById('mAcc').textContent    = m.accuracy;
}


/* ════════════════════════════════════
   Module 2 — Load analytics charts
════════════════════════════════════ */
async function loadCharts() {
  const res  = await fetch('/api/analytics-data');
  const json = await res.json();
  const c    = json.charts;

  /* 1. Attrition by Department */
  barChart('chartDept',
    c.attrition_by_dept.labels,
    c.attrition_by_dept.values,
    [C.rose, C.amber, C.indigo]
  );

  /* 2. Attrition by Gender */
  donutChart('chartGender',
    c.attrition_by_gender.labels,
    c.attrition_by_gender.values,
    [C.indigo, C.rose]
  );

  /* 3. Attrition by Overtime */
  donutChart('chartOvertime',
    c.attrition_by_overtime.labels,
    c.attrition_by_overtime.values,
    [C.emerald, C.rose]
  );

  /* 4. Attrition by Job Role (horizontal) */
  barChart('chartRole',
    c.attrition_by_role.labels,
    c.attrition_by_role.values,
    C.indigo,
    true
  );

  /* 5. Attrition by Age Group */
  barChart('chartAge',
    c.attrition_by_age.labels,
    c.attrition_by_age.values,
    [C.rose, C.amber, C.indigo, C.cyan, C.emerald]
  );

  /* 6. Attrition by Education */
  barChart('chartEdu',
    c.attrition_by_education.labels,
    c.attrition_by_education.values,
    C.purple
  );

  /* 7. Work-Life Balance (line chart) */
  new Chart(document.getElementById('chartWlb'), {
    type: 'line',
    data: {
      labels  : c.worklife_balance.labels,
      datasets: [{
        data               : c.worklife_balance.values,
        borderColor        : B.emerald,
        backgroundColor    : 'rgba(16,185,129,.1)',
        borderWidth        : 2.5,
        pointBackgroundColor: B.emerald,
        pointRadius        : 5,
        fill               : true,
        tension            : .3,
      }],
    },
    options: { ...baseOpts(), scales: scaleOpts },
  });

  /* 8. Monthly Income — Stay vs Leave */
  new Chart(document.getElementById('chartIncome'), {
    type: 'bar',
    data: {
      labels  : c.income_distribution.labels,
      datasets: [{
        data           : c.income_distribution.values,
        backgroundColor: [C.emerald, C.rose],
        borderRadius   : 8,
        borderSkipped  : false,
      }],
    },
    options: {
      ...baseOpts(),
      scales: {
        x: scaleOpts.x,
        y: {
          ...scaleOpts.y,
          ticks: {
            color   : '#94A3B8',
            callback: v => '₹' + v.toLocaleString('en-IN'),
          },
        },
      },
    },
  });

  /* 9. Correlation heatmap (horizontal bar, colour-coded) */
  const cv = c.correlation.values;
  new Chart(document.getElementById('chartCorr'), {
    type: 'bar',
    data: {
      labels  : c.correlation.labels,
      datasets: [{
        data           : cv,
        backgroundColor: cv.map(v =>
          v > 0
            ? `rgba(244,63,94,${Math.min(Math.abs(v) * 4, .9)})`
            : `rgba(16,185,129,${Math.min(Math.abs(v) * 4, .9)})`
        ),
        borderRadius  : 4,
        borderSkipped : false,
      }],
    },
    options: {
      ...baseOpts(),
      indexAxis: 'y',
      scales: {
        x: { ...scaleOpts.x, min: -.25, max: .25 },
        y: { ...scaleOpts.y, grid: { color: 'transparent' } },
      },
    },
  });
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
  loadDashboardStats();
  loadCharts();
});