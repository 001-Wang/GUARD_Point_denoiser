const sourceBase = 'https://guard-point-cloud-zuoxu.zuoxu.chatgpt.site/assets/visualization/';
const localBase = 'assets/visualization/';
const samples = [0, 1, 2, 3];
const heroSample = document.querySelector('#hero-sample');
heroSample?.addEventListener('change', (event) => {
  const n = samples[Number(event.target.value)];
  const base = n === 0 ? localBase : sourceBase;
  document.querySelector('#hero-original').src = `${base}HDD${n}_0.png`;
  document.querySelector('#hero-cvar').src = `${base}HDD${n}_2.png`;
  document.querySelector('#hero-guard').src = `${base}HDD${n}_1.png`;
});

const benchmarkData = {
  hdd: { labels: ['Vanilla', 'PointCVaR', 'GUARD'], values: [0.7739, 0.6685, 0.8318], note: 'GUARD improves PointNet++ by 5.79 percentage points over the vanilla backbone in this evaluation.' },
  scannet: { labels: ['Vanilla', 'PointCVaR', 'GUARD'], values: [0.7757, 0.7521, 0.7902], note: 'GUARD adds 1.45 percentage points on ScanNet with Point Transformer V3.' },
  shape: { labels: ['Vanilla', 'PointCVaR', 'GUARD'], values: [0.6841, 0.6612, 0.7203], note: 'ShapeNetPart average over the reported synthetic corruption evaluations.' },
  ghost: { labels: ['Vanilla', 'PointCVaR', 'GUARD'], values: [0.6761, 0.6534, 0.7048], note: 'GUARD remains robust under GhostCluster corruption at 40% noise.' },
  global: { labels: ['Vanilla', 'PointCVaR', 'GUARD'], values: [0.6943, 0.6719, 0.7162], note: 'Global corruption results show improved segmentation after uncertainty-aware filtering.' },
  local: { labels: ['Vanilla', 'PointCVaR', 'GUARD'], values: [0.7025, 0.6847, 0.7281], note: 'Local corruption results show the same trend across the three methods.' }
};
function renderChart() {
  const key = document.querySelector('#benchmark')?.value || 'hdd';
  const data = benchmarkData[key];
  document.querySelector('#benchmark-chart').innerHTML = data.values.map((value, index) => `<div class="bar-group"><span class="bar-value">${value.toFixed(4)}</span><div class="bar ${index === 0 ? 'vanilla' : index === 1 ? 'pointcvar' : ''}" style="--height:${value * 190}px"></div><span class="bar-label">${data.labels[index]}</span></div>`).join('');
  document.querySelector('#chart-note').textContent = data.note;
}
document.querySelector('#benchmark')?.addEventListener('change', renderChart);
document.querySelector('#backbone')?.addEventListener('change', renderChart);
renderChart();

document.querySelectorAll('.disclosure').forEach((button) => button.addEventListener('click', () => {
  const body = button.nextElementSibling;
  const open = button.getAttribute('aria-expanded') === 'true';
  button.setAttribute('aria-expanded', String(!open));
  button.querySelector('span').textContent = open ? '+' : '−';
  body.hidden = open;
}));

document.querySelector('#copy-citation')?.addEventListener('click', async (event) => {
  const text = document.querySelector('#citation-text').textContent;
  try { await navigator.clipboard.writeText(text); event.currentTarget.textContent = 'Copied ✓'; setTimeout(() => { event.currentTarget.textContent = 'Copy citation'; }, 1600); } catch { event.currentTarget.textContent = 'Select citation above'; }
});
