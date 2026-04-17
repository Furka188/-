async function postForm(url, form) {
  const body = new FormData(form);
  const response = await fetch(url, { method: 'POST', body });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Unknown error');
  return data;
}

function metricBlock(title, payload) {
  return `
    <h3>${title}</h3>
    <pre>${JSON.stringify(payload, null, 2)}</pre>
  `;
}

function linkList(links) {
  return `
    <div class="links">
      ${links.map((link) => `<a href="${link.href}" target="_blank" rel="noopener noreferrer">${link.label}</a>`).join('')}
    </div>
  `;
}

function renderAnalyze(data) {
  return `
    ${metricBlock('Metric Topology', data.metrics)}
    ${metricBlock('Image Quality Envelope', data.quality)}
    <h3>Landmark Volume</h3><pre>${data.landmarks_count}</pre>
    ${linkList([
      { label: 'Скачать фото с биометрией', href: data.overlay_download },
      { label: 'Скачать TXT досье', href: data.report_download },
    ])}
  `;
}

function renderCompare(data) {
  return `
    ${metricBlock('Metric Differences', data.comparison.metric_differences)}
    ${metricBlock('Top Distinguishing Landmarks', data.comparison.top_distinguishing_landmarks)}
    ${linkList([
      { label: 'Overlay фото A', href: data.first_overlay },
      { label: 'Overlay фото B', href: data.second_overlay },
      { label: 'Отчёт A (TXT)', href: data.first_report },
      { label: 'Отчёт B (TXT)', href: data.second_report },
    ])}
  `;
}

const analyzeForm = document.getElementById('analyzeForm');
const analyzeOutput = document.getElementById('analyzeOutput');

analyzeForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  analyzeOutput.innerHTML = '<h3>Status</h3><pre>Обработка...</pre>';
  try {
    const data = await postForm('/analyze', analyzeForm);
    analyzeOutput.innerHTML = renderAnalyze(data);
  } catch (error) {
    analyzeOutput.innerHTML = `<h3>Error</h3><pre>${error.message}</pre>`;
  }
});

const compareForm = document.getElementById('compareForm');
const compareOutput = document.getElementById('compareOutput');

compareForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  compareOutput.innerHTML = '<h3>Status</h3><pre>Обработка...</pre>';
  try {
    const data = await postForm('/compare', compareForm);
    compareOutput.innerHTML = renderCompare(data);
  } catch (error) {
    compareOutput.innerHTML = `<h3>Error</h3><pre>${error.message}</pre>`;
  }
});
