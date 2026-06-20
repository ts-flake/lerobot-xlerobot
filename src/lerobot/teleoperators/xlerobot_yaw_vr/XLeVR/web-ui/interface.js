/**
 * XLeVR interface.js
 * Desktop UI: status polling, settings modal, robot engagement, camera feed toggle.
 */

// ── State ──────────────────────────────────────────────────────────────────

let robotEngaged = false;
let keyboardEnabled = false;
let cameraFeedActive = false;
let cameraFeedUrl = '';

// ── Init ───────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  detectDeviceType();
  loadConfig();
  startStatusPolling();

  // Settings form submit
  const form = document.getElementById('settingsForm');
  if (form) form.addEventListener('submit', saveConfig);
});

function detectDeviceType() {
  const isVR = /OculusBrowser|Quest/i.test(navigator.userAgent);
  const desktop = document.getElementById('desktopInterface');
  const vrContent = document.getElementById('vrContent');
  if (desktop)  desktop.style.display  = isVR ? 'none'  : 'block';
  if (vrContent) vrContent.style.display = isVR ? 'block' : 'none';
}

// ── Settings modal ─────────────────────────────────────────────────────────

function openSettings() {
  const modal = document.getElementById('settingsModal');
  if (modal) modal.style.display = 'flex';
}

function closeSettings() {
  const modal = document.getElementById('settingsModal');
  if (modal) modal.style.display = 'none';
}

function loadConfig() {
  fetch('/api/config')
    .then(r => r.ok ? r.json() : null)
    .then(cfg => {
      if (!cfg) return;
      _setVal('httpsPort',      cfg.network?.https_port      ?? '');
      _setVal('websocketPort',  cfg.network?.websocket_port  ?? '');
      _setVal('vrScale',        cfg.robot?.vr_to_robot_scale ?? '');
      _setVal('gripRake',       cfg.grip_rake_deg            ?? '');
      _setVal('gripYaw',        cfg.grip_yaw_deg             ?? '');
      cameraFeedUrl = cfg.camera_feed_url ?? '';
      _setVal('cameraFeedUrl',  cameraFeedUrl);
    })
    .catch(() => { /* server may not have /api/config — ignore */ });
}

function saveConfig(evt) {
  evt.preventDefault();
  const cfg = {
    network: {
      https_port:     parseInt(_getVal('httpsPort'),     10) || undefined,
      websocket_port: parseInt(_getVal('websocketPort'), 10) || undefined,
    },
    robot: {
      vr_to_robot_scale: parseFloat(_getVal('vrScale')) || undefined,
    },
    grip_rake_deg: parseFloat(_getVal('gripRake')) || undefined,
    grip_yaw_deg:  parseFloat(_getVal('gripYaw'))  || undefined,
    camera_feed_url: _getVal('cameraFeedUrl'),
  };

  // Update local camera URL immediately
  cameraFeedUrl = cfg.camera_feed_url;
  if (cameraFeedActive) {
    const img = document.getElementById('cameraFeedImg');
    if (img) img.src = cameraFeedUrl;
  }

  fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  })
    .then(() => closeSettings())
    .catch(() => closeSettings());
}

// ── Status polling ─────────────────────────────────────────────────────────

function startStatusPolling() {
  fetchStatus();
  setInterval(fetchStatus, 3000);
}

function fetchStatus() {
  fetch('/api/status')
    .then(r => r.ok ? r.json() : null)
    .then(s => { if (s) updateStatus(s); })
    .catch(() => {});
}

function updateStatus(s) {
  _setIndicator('leftArmStatus',  s.leftArm  ?? s.left_arm);
  _setIndicator('rightArmStatus', s.rightArm ?? s.right_arm);
  _setIndicator('vrStatus',       s.vrConnected ?? s.vr_connected);
  if (s.engaged !== undefined) {
    robotEngaged = !!s.engaged;
    _updateEngageButton();
  }
}

// ── Robot engagement ───────────────────────────────────────────────────────

function toggleRobotEngagement() {
  robotEngaged = !robotEngaged;
  fetch('/api/robot', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ engaged: robotEngaged }),
  }).catch(() => {});
  _updateEngageButton();
}

function _updateEngageButton() {
  const btn = document.getElementById('engageBtnText');
  const status = document.getElementById('engagementStatusText');
  if (btn) btn.textContent = robotEngaged ? '🔴 Disconnect Robot' : '🔌 Connect Robot';
  if (status) status.textContent = robotEngaged ? 'Motors Engaged' : 'Motors Disengaged';
}

// ── Camera feed ────────────────────────────────────────────────────────────

function toggleCameraFeed() {
  const container = document.getElementById('cameraFeedContainer');
  const btn = document.getElementById('cameraFeedToggle');
  const img = document.getElementById('cameraFeedImg');
  if (!container || !img) return;

  cameraFeedActive = !cameraFeedActive;
  if (cameraFeedActive) {
    const url = cameraFeedUrl || _getVal('cameraFeedUrl');
    if (!url) {
      alert('No camera feed URL configured. Open ⚙️ Settings and set a Camera Feed URL.');
      cameraFeedActive = false;
      return;
    }
    img.src = url;
    container.style.display = 'block';
    if (btn) btn.textContent = 'Hide Feed';
  } else {
    img.src = '';   // stop loading stream
    container.style.display = 'none';
    if (btn) btn.textContent = 'Show Feed';
  }
}

// ── Keyboard control ───────────────────────────────────────────────────────

function toggleKeyboardControl() {
  keyboardEnabled = !keyboardEnabled;
  fetch('/api/keyboard', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: keyboardEnabled }),
  }).catch(() => {});

  const toggle = document.getElementById('keyboardToggle');
  const text   = document.getElementById('keyboardToggleText');
  const collapsed = document.querySelector('.keyboard-help-collapsed');
  const expanded  = document.querySelector('.keyboard-help-expanded');

  if (keyboardEnabled) {
    if (collapsed) collapsed.style.display = 'none';
    if (expanded)  expanded.style.display  = 'block';
  } else {
    if (collapsed) collapsed.style.display = 'block';
    if (expanded)  expanded.style.display  = 'none';
  }
}

document.addEventListener('keydown', (evt) => {
  if (!keyboardEnabled) return;
  const key = evt.key.toLowerCase();
  fetch('/api/keypress', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key }),
  }).catch(() => {});
});

// ── Helpers ────────────────────────────────────────────────────────────────

function _getVal(id) {
  const el = document.getElementById(id);
  return el ? el.value.trim() : '';
}

function _setVal(id, val) {
  const el = document.getElementById(id);
  if (el && val !== undefined && val !== null) el.value = val;
}

function _setIndicator(id, connected) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'status-indicator ' + (connected ? 'connected' : 'disconnected');
}
