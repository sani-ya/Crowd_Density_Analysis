// CrowdSense Dashboard — app.js

const state = {
    viewMode: 'feed',
    imagePreview: null,
    isLive: false,
    crowdCount: 0,
    densityLevel: 'none',
    anomaly: false,
    confidence: 0
};

const el = {
    statusBadge: document.getElementById('statusBadge'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),

    fileInput: document.getElementById('fileInput'),
    btnUpload: document.getElementById('btnUpload'),
    btnWebcam: document.getElementById('btnWebcam'),
    webcamIcon: document.getElementById('webcamIcon'),
    webcamText: document.getElementById('webcamText'),

    countCard: document.getElementById('countCard'),
    crowdCount: document.getElementById('crowdCount'),
    crowdSub: document.getElementById('crowdSub'),

    densityCard: document.getElementById('densityCard'),
    densityIcon: document.getElementById('densityIcon'),
    densityBar: document.getElementById('densityBar'),
    densityLabel: document.getElementById('densityLabel'),

    anomalyCard: document.getElementById('anomalyCard'),
    anomalyIcon: document.getElementById('anomalyIcon'),
    anomalyStatus: document.getElementById('anomalyStatus'),
    anomalySub: document.getElementById('anomalySub'),

    alertBadge: document.getElementById('alertBadge'),
    confRingCircle: document.getElementById('confRingCircle'),
    confPct: document.getElementById('confPct'),
    confSub: document.getElementById('confSub')
};

const CIRCUMFERENCE = 163.36;

// ---- STATUS UPDATE ----
function setStatus(text, state) {
    el.statusText.textContent = text;
    const colorMap = {
        ok: { color: '#059669', bg: 'rgba(5,150,105,0.1)', border: 'rgba(5,150,105,0.25)' },
        busy: { color: '#d97706', bg: 'rgba(217,119,6,0.1)', border: 'rgba(217,119,6,0.25)' },
        err: { color: '#dc2626', bg: 'rgba(220,38,38,0.1)', border: 'rgba(220,38,38,0.25)' },
        idle: { color: '#5a6580', bg: '#ffffff', border: 'rgba(46,67,159,0.1)' }
    };
    const c = colorMap[state] || colorMap.idle;
    el.statusDot.style.background = c.color;
    el.statusDot.style.boxShadow = `0 0 6px ${c.color}`;
    el.statusBadge.style.color = c.color;
    el.statusBadge.style.background = c.bg;
    el.statusBadge.style.borderColor = c.border;
}

// ---- CHART ENGINE ----
const ctx = document.getElementById('densityChart').getContext('2d');
const densityChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Total Crowd Count',
            data: [],
            borderColor: '#2e439f',
            backgroundColor: 'rgba(46, 67, 159, 0.15)',
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointRadius: 2,
            pointHoverRadius: 6,
            pointBackgroundColor: '#2e439f'
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300, easing: 'easeOutQuart' },
        scales: {
            x: { display: false },
            y: { 
                beginAtZero: true,
                grid: { color: 'rgba(0,0,0,0.06)' },
                border: { dash: [5, 5] }
            }
        },
        plugins: {
            legend: { display: false },
            tooltip: { backgroundColor: 'rgba(15, 22, 41, 0.9)', titleFont: {family: 'Outfit'}, bodyFont: {family: 'Outfit', size: 14} }
        }
    }
});

let timePoints = 0;
function logChartData(count) {
    const maxPoints = 50; // Keep last 50 data points on screen
    densityChart.data.labels.push(timePoints++);
    densityChart.data.datasets[0].data.push(count);

    if (densityChart.data.labels.length > maxPoints) {
        densityChart.data.labels.shift();
        densityChart.data.datasets[0].data.shift();
    }
    densityChart.update();
}

// ---- METRICS UPDATE ----
function updateMetrics(count, density, anomaly, conf, mode) {
    // Count
    animateCount(el.crowdCount, count);
    if(mode) {
        el.crowdSub.textContent = `Algorithm Mode: ${mode}`;
    } else {
        el.crowdSub.textContent = `Analysis complete`;
    }

    // Density
    const densityPct = Math.min((count / 200) * 100, 100);
    el.densityBar.style.width = densityPct + '%';
    el.densityLabel.textContent = density;

    // Density card class
    el.densityCard.className = 'metric-card glass';
    el.countCard.className = 'metric-card glass';
    if (density === 'Critical') {
        el.densityCard.classList.add('state-critical');
        el.countCard.classList.add('state-critical');
        el.densityIcon.className = 'mc-icon icon-critical';
    } else if (density === 'Crowded') {
        el.densityCard.classList.add('state-crowded');
        el.countCard.classList.add('state-crowded');
        el.densityIcon.className = 'mc-icon icon-crowded';
    } else {
        el.densityCard.classList.add('state-normal');
        el.countCard.classList.add('state-normal');
        el.densityIcon.className = 'mc-icon icon-normal';
    }
    el.densityIcon.innerHTML = '<i class="ph ph-gauge"></i>';

    // Anomaly
    if (anomaly) {
        el.anomalyCard.className = 'metric-card glass state-critical';
        el.anomalyStatus.textContent = 'Anomaly Detected!';
        el.anomalySub.textContent = 'Unusual crowd behavior';
        el.anomalyIcon.className = 'mc-icon icon-critical';
        el.anomalyIcon.innerHTML = '<i class="ph ph-warning"></i>';
        el.alertBadge.style.display = '';
    } else {
        el.anomalyCard.className = 'metric-card glass state-normal';
        el.anomalyStatus.textContent = 'No Anomaly';
        el.anomalySub.textContent = 'System monitoring normally';
        el.anomalyIcon.className = 'mc-icon icon-normal';
        el.anomalyIcon.innerHTML = '<i class="ph ph-shield-check"></i>';
        el.alertBadge.style.display = 'none';
    }

    // Confidence ring
    const pct = Math.round(conf);
    el.confPct.textContent = pct + '%';
    el.confSub.textContent = `CSRNet model confidence`;
    const offset = CIRCUMFERENCE - (CIRCUMFERENCE * pct / 100);
    el.confRingCircle.style.strokeDashoffset = offset;

    // Chart Update
    logChartData(count);
}

function animateCount(el, target) {
    const start = parseInt(el.textContent) || 0;
    const duration = 600;
    const startTime = performance.now();
    function step(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(start + (target - start) * eased);
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

// ---- VIEWER UPDATE ----
function updateViewer() {
    // The live view image has been replaced by the Live Density Chart.
}

// ---- EVENTS ----

// Upload
el.btnUpload.addEventListener('click', () => el.fileInput.click());
el.fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    state.isLive = false;
    
    setStatus('Analyzing Frame...', 'busy');

    const formData = new FormData();
    formData.append('image', file);

    try {
        const res = await fetch('http://localhost:5000/api/analyze_image', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        
        state.imagePreview = data.image; // Replace original with heatmapped response
        updateViewer();
        updateMetrics(data.count, data.density, data.anomaly, data.conf, data.mode);
        setStatus('Analysis Complete', 'ok');
    } catch(err) {
        setStatus('Backend Connection Error', 'err');
    }

    e.target.value = '';
});

// Webcam
el.btnWebcam.addEventListener('click', () => {
    if (!state.isLive) {
        state.isLive = true;
        state.imagePreview = 'http://localhost:5000/api/video_feed';
        el.btnWebcam.classList.add('active');
        el.webcamIcon.className = 'ph ph-stop-circle';
        el.webcamText.textContent = 'Stop Stream';
        updateViewer();
        setStatus('Connecting to Webcam Backend...', 'busy');

        setTimeout(() => { 
            setStatus('Live Stream Active', 'ok'); 
        }, 1200);

    } else {
        state.isLive = false;
        fetch('http://localhost:5000/api/stop_feed', { method: 'POST' });
        el.btnWebcam.classList.remove('active');
        el.webcamIcon.className = 'ph ph-video-camera';
        el.webcamText.textContent = 'Start Webcam';
        updateViewer();
        setStatus('Ready to Analyze', 'idle');
    }
});

// Init
setStatus('Ready to Analyze', 'idle');

// Global Polling Loop: continuously updates the dash from the Python backend
let lastTimestamp = 0;
setInterval(async () => {
    try {
        const res = await fetch('http://localhost:5000/api/stats');
        const data = await res.json();
        if (data.timestamp && data.timestamp > lastTimestamp) {
            lastTimestamp = data.timestamp;
            updateMetrics(data.count, data.density, data.anomaly, data.conf, data.mode);
        }
    } catch (e) { } // silent fail if backend offline
}, 800);
