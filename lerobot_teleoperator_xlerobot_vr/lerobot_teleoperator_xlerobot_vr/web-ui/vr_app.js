/**
 * XLeVR vr_app.js
 *
 * Streams controller + headset poses (and the operator's chest-yaw reference)
 * to the Python WS server.
 *
 * Naming: R_a_b means "frame b expressed in a coords". All expressed using quaternions.
 * Composition reads right-to-left:  R_a_c = R_a_b · R_b_c.
 *
 * Frames:
 *   vr      WebXR local-floor (+X right, +Y up, −Z forward; OpenGL)
 *   rb      Robot/World frame (+X forward, +Y left, +Z up); pure axis swap of vr
 *   hmd     Head-mounted display (+X right, +Y up, −Z forward; OpenGL); same as vr
 *   head    Head (+X forward, +Y left, +Z up); pure axis swap of hmd
 *   g[h]    Per-hand controller grip (raw WebXR)
 *   ee[h]   Per-hand robot EE (g rotated by rake + ±yaw)
 *   chest   Operator yaw reference: rb rotated about rb +Z by chest_yaw_rad.
 *           Frozen by default; B toggles freeze. EMA-smoothed when unfrozen.
 *           Downstream uses R_rb_chest to interpret pos deltas in the
 *           operator's "forward" sense regardless of head look direction.
 */

const XR_BUTTONS = {
  TRIGGER: 0, SQUEEZE: 1,
  THUMBSTICK_CLICK: 3, PRIMARY: 4, SECONDARY: 5,
};

// In-VR camera panel layout
const CAMERA_LAYOUT = {
  order:       ['left_wrist', 'head', 'right_wrist'], // left→right; cams not listed are appended
  y:           1.25,   // panel centre height above the floor (m)   ← adjust to raise/lower
  distance:    1.2,   // forward distance from the operator (m)
  spacing:     0.7,   // horizontal gap between panels (m)
  width:       0.6,   // panel width (m)
  height:      0.45,  // panel height (m)
  followChest: true,  // panels follow the operator's chest yaw + position (re-orient with the body)
};

AFRAME.registerComponent('controller-updater', {
  init: function () {
    this.leftHand  = document.querySelector('#leftHand');
    this.rightHand = document.querySelector('#rightHand');
    this.headset   = document.querySelector('#headset');
    this.leftInfo  = document.querySelector('#leftHandInfo');
    this.rightInfo = document.querySelector('#rightHandInfo');
    this.headInfo  = document.querySelector('#headsetInfo');

    this.leftTriggerDown  = false;
    this.rightTriggerDown = false;
    this._prevBPressed    = false;

    // Chest yaw — frozen by default
    this.chestFrozen   = true;
    this.chestYaw_rad  = 0;
    this.chestEmaAlpha = 0.9;

    // Viz toggles
    this.showRefFrames = false;
    this.rbFrameEl     = null;
    this.chestFrameEl  = null;

    // Grip correction (default until server pushes xlevr_config)
    this.gripRake_deg = 40;
    this.gripYaw_deg  = 10;

    // R_rb_vr: constant axis swap. Columns = vr basis in rb.
    //   R_rb_vr = [[0,0,−1],[−1,0,0],[0,1,0]]
    const _m = new THREE.Matrix4().set(0,0,-1,0, -1,0,0,0, 0,1,0,0, 0,0,0,1);
    this.R_rb_vr = new THREE.Quaternion().setFromRotationMatrix(_m);

    // R_ee_g[hand]: per-hand g→ee (rake + ±yaw)
    this._recomputeREEg();

    this._buildHandAxes(this.leftHand,  'left');
    this._buildHandAxes(this.rightHand, 'right');
    this._buildRbFrame();
    this._buildChestFrame();

    const tRot = '-90 0 0';
    if (this.leftInfo)  this.leftInfo.setAttribute('rotation', tRot);
    if (this.rightInfo) this.rightInfo.setAttribute('rotation', tRot);

    this._connectWS();

    // ── Camera overlay: auto-show every available feed (capped at maxFeeds) ──
    // No in-VR interaction (no menu/raycaster/buttons) — the controllers are
    // fully owned by teleop. JPEG frames are decoded off the main thread.
    this.camPanels    = document.querySelector('#camPanels');
    this.maxFeeds     = 3;           // bandwidth cap on simultaneous feeds
    this.camEntries   = {};          // name -> { canvas, ctx, texture, plane, pending }
    this.shownCameras = [];
    this._panelsVisible = true;      // toggled by right thumbstick click
    this._overlayDisabled = false;   // guard: never let the overlay break the render loop

    this._addControllerListeners(this.leftHand,  'left');
    this._addControllerListeners(this.rightHand, 'right');
    if (this.leftHand) {
      this.leftHand.addEventListener('thumbstickdown', () => this._toggleRefFrames());
    }
    // Right thumbstick click (free — teleop uses the axes, not the click) toggles
    // the camera feeds on/off. Hiding also unsubscribes, so it costs nothing.
    if (this.rightHand) {
      this.rightHand.addEventListener('thumbstickdown', () => this._toggleCamPanels());
    }
  },

  // ── Static rotations ─────────────────────────────────────────────────────

  _computeREEg: function (rake_deg, yaw_deg) {
    const t = rake_deg * Math.PI / 180;
    const m_rake = new THREE.Matrix4().set(
       0, -Math.sin(t), -Math.cos(t), 0,
      -1,  0,            0,           0,
       0,  Math.cos(t), -Math.sin(t), 0,
       0,  0,            0,           1
    );
    const m_yaw = new THREE.Matrix4().makeRotationZ(yaw_deg * Math.PI / 180);
    const m = new THREE.Matrix4().multiplyMatrices(m_yaw, m_rake);
    return new THREE.Quaternion().setFromRotationMatrix(m);
  },

  _recomputeREEg: function () {
    this.R_ee_g = {
      left:  this._computeREEg(this.gripRake_deg, +this.gripYaw_deg),
      right: this._computeREEg(this.gripRake_deg, -this.gripYaw_deg),
    };
  },

  // ── Visualization ────────────────────────────────────────────────────────

  _eeAxesInG: function (R_ee_g) {
    const q = R_ee_g.clone().conjugate();
    return [
      new THREE.Vector3(1, 0, 0).applyQuaternion(q),
      new THREE.Vector3(0, 1, 0).applyQuaternion(q),
      new THREE.Vector3(0, 0, 1).applyQuaternion(q),
    ];
  },

  _buildHandAxes: function (handEl, side) {
    if (!handEl) return;
    const dirs = this._eeAxesInG(this.R_ee_g[side]);
    const colors = ['#ff0000', '#00ff00', '#0000ff'];
    const L = 0.08, r = 0.002;
    const container = document.createElement('a-entity');
    container.setAttribute('class', 'ee-axes');
    dirs.forEach((d, i) => this._appendArrow(container, colors[i], d, L, r));
    handEl.appendChild(container);
  },

  _rebuildHandAxes: function (handEl, side) {
    if (!handEl) return;
    const old = handEl.querySelector('.ee-axes');
    if (old) old.parentNode.removeChild(old);
    this._buildHandAxes(handEl, side);
  },

  _buildAxesTriad: function (id, label, length, radius) {
    // Local-frame triad (+X red, +Y green, +Z blue). Caller sets parent transform.
    const f = document.createElement('a-entity');
    f.setAttribute('id', id);
    f.setAttribute('visible', 'false');
    const dirs = [
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 0, 1),
    ];
    const colors = ['#ff0000', '#00ff00', '#0000ff'];
    dirs.forEach((d, i) => this._appendArrow(f, colors[i], d, length, radius));
    this._appendLabel(f, label, -length * .2);
    return f;
  },

  _buildRbFrame: function () {
    const scene = document.querySelector('a-scene');
    const f = this._buildAxesTriad('rbFrame', 'world', 0.2, 0.002);
    scene.appendChild(f);
    // R_vr_rb = R_rb_vr⁻¹, so triad local axes land at correct vr orientations.
    f.object3D.quaternion.copy(this.R_rb_vr).conjugate();
    this.rbFrameEl = f;
  },

  _buildChestFrame: function () {
    // Position + orientation updated each tick from hmd + chestYaw_rad.
    const scene = document.querySelector('a-scene');
    const f = this._buildAxesTriad('chestFrame', 'chest', 0.10, 0.002);
    scene.appendChild(f);
    this.chestFrameEl = f;
    
  },

  _appendArrow: function (parentEl, color, dir, length, radius) {
    const fmt = (v) => v.toFixed(4);
    const H = length / 2;
    const pShaft = dir.clone().multiplyScalar(H);
    const pTip   = dir.clone().multiplyScalar(length);
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    const e = new THREE.Euler().setFromQuaternion(q, 'YXZ');  // A-Frame's order
    const rot = `${fmt(e.x*180/Math.PI)} ${fmt(e.y*180/Math.PI)} ${fmt(e.z*180/Math.PI)}`;

    const shaft = document.createElement('a-cylinder');
    shaft.setAttribute('height',   String(length));
    shaft.setAttribute('radius',   String(radius));
    shaft.setAttribute('position', `${fmt(pShaft.x)} ${fmt(pShaft.y)} ${fmt(pShaft.z)}`);
    shaft.setAttribute('rotation', rot);
    shaft.setAttribute('material', `shader: flat; color: ${color}`);
    parentEl.appendChild(shaft);

    const tip = document.createElement('a-cone');
    tip.setAttribute('height',        String(length * 0.18));
    tip.setAttribute('radius-bottom', String(radius * 3));
    tip.setAttribute('radius-top',    '0');
    tip.setAttribute('position', `${fmt(pTip.x)} ${fmt(pTip.y)} ${fmt(pTip.z)}`);
    tip.setAttribute('rotation', rot);
    tip.setAttribute('material', `shader: flat; color: ${color}`);
    parentEl.appendChild(tip);
  },

  _appendLabel: function (parentEl, text, xOffset) {
    const label = document.createElement('a-text');
    label.setAttribute('value', text);
    label.setAttribute('position', `${xOffset.toFixed(3)} 0 0`);
    label.setAttribute('rotation', '0 0 -90')
    label.setAttribute('scale', '0.08 0.08 0.08');
    label.setAttribute('align', 'center');
    label.setAttribute('color', '#ffffff');
    label.setAttribute('side', 'double');
    parentEl.appendChild(label);
  },

  // ── Headset frame mapping ────────────────────────────────────────────────

  /** R_rb_hmd = R_rb_vr · R_vr_hmd  (hmd: head-mounted display in OpenGL convention). */
  _R_rb_hmd: function () {
    const R_vr_hmd = this.headset.object3D.quaternion;
    return new THREE.Quaternion().multiplyQuaternions(this.R_rb_vr, R_vr_hmd);
  },

  /** R_rb_head = R_rb_vr · R_vr_hmd · R_hmd_head, where R_hmd_head = R_rb_vr⁻¹
   */
  _R_rb_head: function () {
    return new THREE.Quaternion()
      .multiplyQuaternions(this.R_rb_vr, this.headset.object3D.quaternion)
      .multiply(this.R_rb_vr.clone().conjugate());
  },

  /** Head's forward direction (local hmd -Z) expressed in rb coords. */
  _headForward_rb: function () {
    return new THREE.Vector3(0, 0, -1).applyQuaternion(this._R_rb_hmd());
  },

  // ── Chest yaw ────────────────────────────────────────────────────────────

  _snapChestYawToHead: function () {
    if (!this.headset || !this.headset.object3D) return;
    const fwd = this._headForward_rb();
    this.chestYaw_rad = Math.atan2(fwd.y, fwd.x);
  },

  _toggleChestFrozen: function () {
    this.chestFrozen = !this.chestFrozen;
    if (!this.chestFrozen) this._snapChestYawToHead();  // snap on freeze→follow edge
    console.log(`[chest] frozen=${this.chestFrozen} yaw=${(this.chestYaw_rad * 180 / Math.PI).toFixed(1)}°`);
  },

  _updateChestYaw: function () {
    if (this.chestFrozen) return;
    if (!this.headset || !this.headset.object3D) return;
    const fwd = this._headForward_rb();
    const targetYaw = Math.atan2(fwd.y, fwd.x);
    const da = this._wrapPi(targetYaw - this.chestYaw_rad);
    this.chestYaw_rad += this.chestEmaAlpha * da;
  },

  _wrapPi: function (a) { return Math.atan2(Math.sin(a), Math.cos(a)); },

  _chestQuat_rb: function () {
    return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), this.chestYaw_rad);
  },

  _updateChestFrameTransform: function () {
    if (!this.chestFrameEl || !this.headset || !this.headset.object3D) return;
    // R_vr_chest = R_vr_rb · R_rb_chest
    const R_vr_rb    = this.R_rb_vr.clone().conjugate();
    const R_rb_chest = this._chestQuat_rb();
    const R_vr_chest = new THREE.Quaternion().multiplyQuaternions(R_vr_rb, R_rb_chest);
    this.chestFrameEl.object3D.quaternion.copy(R_vr_chest);
    // Position: 35 cm below head (world down = vr −Y) + 15 cm along chest +X
    const hmd_pos_vr = this.headset.object3D.position;
    const chest_pos_vr = new THREE.Vector3(0.15, 0, -0.35).applyQuaternion(R_vr_chest);
    this.chestFrameEl.object3D.position.set(
      hmd_pos_vr.x + chest_pos_vr.x,
      hmd_pos_vr.y + chest_pos_vr.y,
      hmd_pos_vr.z + chest_pos_vr.z,
    );
    // Dynamic label: chest [FROZEN|FOLLOW]
    const labelEl = this.chestFrameEl.querySelector('a-text');
    if (labelEl) {
      const want = `chest [${this.chestFrozen ? 'FROZEN' : 'FOLLOW'}]`;
      if (labelEl.getAttribute('value') !== want) labelEl.setAttribute('value', want);
    }
  },

  // ── UI ───────────────────────────────────────────────────────────────────

  _toggleRefFrames: function () {
    this.showRefFrames = !this.showRefFrames;
    const v = String(this.showRefFrames);
    if (this.rbFrameEl)    this.rbFrameEl.setAttribute('visible', v);
    if (this.chestFrameEl) this.chestFrameEl.setAttribute('visible', v);
  },

  _addControllerListeners: function (handEl, side) {
    if (!handEl) return;
    handEl.addEventListener('triggerdown', () => {
      if (side === 'left') this.leftTriggerDown = true; else this.rightTriggerDown = true;
    });
    handEl.addEventListener('triggerup', () => {
      if (side === 'left') this.leftTriggerDown = false; else this.rightTriggerDown = false;
    });
  },

  // ── WebSocket ────────────────────────────────────────────────────────────

  _connectWS: function () {
    const url = `wss://${window.location.hostname}:8442`;
    try {
      this.ws = new WebSocket(url);
      this.ws.onopen  = () => { if (typeof updateStatus === 'function') updateStatus({ vrConnected: true }); };
      this.ws.onerror = () => { if (typeof updateStatus === 'function') updateStatus({ vrConnected: false }); };
      this.ws.onclose = () => { this.ws = null; if (typeof updateStatus === 'function') updateStatus({ vrConnected: false }); };
      this.ws.onmessage = (evt) => {
        let m;
        try { m = JSON.parse(evt.data); } catch (_) { return; }
        try {
          if      (m.type === 'xlevr_config')  this._onServerConfig(m);
          else if (m.type === 'camera_list')   this._onCameraList(m);
          else if (m.type === 'camera_frame')  this._onCameraFrame(m);
        } catch (e) {
          console.error('ws message handler error:', m && m.type, e);
        }
      };
    } catch (e) { console.error('WS init failed:', e); }
  },

  _onServerConfig: function (msg) {
    if (msg.grip_rake_deg != null) this.gripRake_deg = msg.grip_rake_deg;
    if (msg.grip_yaw_deg  != null) this.gripYaw_deg  = msg.grip_yaw_deg;
    this._recomputeREEg();
    this._rebuildHandAxes(this.leftHand,  'left');
    this._rebuildHandAxes(this.rightHand, 'right');
  },

  // ── Camera overlay ─────────────────────────────────────────────────────────

  _onCameraList: function (msg) {
    // Show every available camera in the configured order, capped at maxFeeds.
    const all  = Array.isArray(msg.cameras) ? msg.cameras : [];
    const cams = this._orderCameras(all).slice(0, this.maxFeeds);
    this.shownCameras = cams;
    this._layoutPanels(cams);                                  // create panels first…
    this._sendSubscription(this._panelsVisible ? cams : []);   // …then stream (unless hidden)
  },

  // Sort camera names by CAMERA_LAYOUT.order (left→right); unlisted names appended.
  _orderCameras: function (cams) {
    const order = CAMERA_LAYOUT.order || [];
    const known = order.filter((n) => cams.includes(n));
    const rest  = cams.filter((n) => !order.includes(n));
    return known.concat(rest);
  },

  // Right-thumbstick toggle: show/hide feeds. Hiding unsubscribes (zero cost).
  _toggleCamPanels: function () {
    this._panelsVisible = !this._panelsVisible;
    if (this.camPanels) this.camPanels.setAttribute('visible', this._panelsVisible);
    this._sendSubscription(this._panelsVisible ? this.shownCameras : []);
  },

  _onCameraFrame: function (msg) {
    const e = this.camEntries[msg.name];
    if (!e) return;                  // only decode for laid-out panels
    // Decode JPEG OFF the main thread; the bitmap is consumed in tick().
    const bytes = Uint8Array.from(atob(msg.data), (c) => c.charCodeAt(0));
    const blob = new Blob([bytes], { type: 'image/jpeg' });
    createImageBitmap(blob).then((bmp) => {
      if (e.pending && e.pending.close) e.pending.close();  // drop unconsumed frame
      e.pending = bmp;
    }).catch(() => {});
  },

  _sendSubscription: function (cams) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ type: 'camera_subscribe', cameras: cams }));
  },

  _removeEntry: function (name) {
    const e = this.camEntries[name];
    if (!e) return;
    if (e.plane && e.plane.parentNode) e.plane.parentNode.removeChild(e.plane);
    if (e.pending && e.pending.close) e.pending.close();
    if (e.texture) e.texture.dispose();
    delete this.camEntries[name];
  },

  // NPOT-safe canvas texture: camera frames are non-power-of-two; mipmaps on
  // NPOT fail to upload (black), so disable them. Linear filter + clamp + SRGB.
  _makeCanvasTexture: function (canvas) {
    const t = new THREE.CanvasTexture(canvas);
    t.generateMipmaps = false;
    t.minFilter = THREE.LinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.wrapS = THREE.ClampToEdgeWrapping;
    t.wrapT = THREE.ClampToEdgeWrapping;
    if ('SRGBColorSpace' in THREE) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  },

  _layoutPanels: function (cams) {
    if (!this.camPanels) return;
    // Drop panels for cameras no longer shown.
    Object.keys(this.camEntries).forEach((name) => {
      if (!cams.includes(name)) this._removeEntry(name);
    });
    const L = CAMERA_LAYOUT;
    const n = cams.length;
    cams.forEach((name, i) => {
      let e = this.camEntries[name];
      if (!e) {
        const canvas = document.createElement('canvas');
        canvas.width = 2; canvas.height = 2;   // placeholder until first frame
        const ctx = canvas.getContext('2d');
        const texture = this._makeCanvasTexture(canvas);
        const plane = document.createElement('a-plane');
        plane.setAttribute('width', String(L.width));
        plane.setAttribute('height', String(L.height));
        plane.setAttribute('material', 'shader: flat; side: double');
        plane.addEventListener('loaded', () => {
          const mesh = plane.getObject3D('mesh');
          if (mesh) {
            mesh.material.map = this.camEntries[name].texture;  // may have been recreated
            mesh.material.side = THREE.DoubleSide;
            mesh.material.needsUpdate = true;
          }
        });
        this.camPanels.appendChild(plane);
        e = { canvas, ctx, texture, plane, pending: null };
        this.camEntries[name] = e;
      }
      const x = (i - (n - 1) / 2) * L.spacing;
      // Local coords: when followChest is on, camPanels is moved/yawed to the
      // operator each tick, so -Z is "in front" and +Z faces the operator.
      e.plane.setAttribute('position', `${x} ${L.y} ${-L.distance}`);
    });
  },

  // Move the whole panel group to the operator's chest frame (position + yaw),
  // so the feeds re-orient with the body. Yaw-only keeps the panels upright and
  // facing the operator; height stays at CAMERA_LAYOUT.y.
  _updateCamPanelsTransform: function () {
    if (!CAMERA_LAYOUT.followChest) return;
    if (!this.camPanels || !this.camPanels.object3D || !this.headset || !this.headset.object3D) return;
    const hp = this.headset.object3D.position;
    this.camPanels.object3D.position.set(hp.x, 0, hp.z);
    this.camPanels.object3D.rotation.set(0, this.chestYaw_rad, 0);
  },

  // ── Per-frame tick ───────────────────────────────────────────────────────

  tick: function () {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    this._updateChestYaw();
    this._updateChestFrameTransform();
    this._updateCamPanelsTransform();
    this._updateCameraOverlay();
    // if (this.showRefFrames) this._billboardLabels();

    const left  = this._collectController(this.leftHand,  this.leftTriggerDown,  'left');
    const right = this._collectController(this.rightHand, this.rightTriggerDown, 'right');
    const head  = this._collectHeadset();

    // B-edge on right controller → toggle chest freeze
    const bNow = !!(right.buttons && right.buttons.b);
    if (bNow && !this._prevBPressed) this._toggleChestFrozen();
    this._prevBPressed = bNow;

    if (this.leftInfo  && left.position)  this.leftInfo.setAttribute('value',  this._poseText('ee_left',  left));
    if (this.rightInfo && right.position) this.rightInfo.setAttribute('value', this._poseText('ee_right', right));
    if (this.headInfo  && head.position)  this.headInfo.setAttribute('value',  this._poseText('headset',  head));

    if (!left.position && !right.position && !head.position) return;

    const q_rb_chest = this._chestQuat_rb();
    this.ws.send(JSON.stringify({
      timestamp:       Date.now(),
      leftController:  left,
      rightController: right,
      headset:         head,
      chestQuaternion: { x: q_rb_chest.x, y: q_rb_chest.y, z: q_rb_chest.z, w: q_rb_chest.w },
      chestFrozen:     this.chestFrozen,
    }));
  },

  _updateCameraOverlay: function () {
    if (this._overlayDisabled) return;
    try {
      // Upload any freshly-decoded frame to its panel texture (cheap blit).
      for (const name in this.camEntries) {
        const e = this.camEntries[name];
        if (!e.pending) continue;
        const bmp = e.pending; e.pending = null;
        const resized = (e.canvas.width !== bmp.width || e.canvas.height !== bmp.height);
        if (resized) { e.canvas.width = bmp.width; e.canvas.height = bmp.height; }
        e.ctx.drawImage(bmp, 0, 0);
        if (bmp.close) bmp.close();
        if (resized) {
          // In-place canvas resize doesn't reliably re-upload to the GPU (panel
          // stays black). Recreate the texture at the real size and rebind it.
          if (e.texture) e.texture.dispose();
          e.texture = this._makeCanvasTexture(e.canvas);
          const mesh = e.plane.getObject3D('mesh');
          if (mesh) { mesh.material.map = e.texture; mesh.material.needsUpdate = true; }
        }
        e.texture.needsUpdate = true;
      }
    } catch (err) {
      // Never let the overlay break the render loop; disable it on error.
      this._overlayDisabled = true;
      console.error('camera overlay disabled:', err);
    }
  },

  _poseText: function (title, ctrl) {
    if (!ctrl.position) return `${title}\n...`;
    const p = ctrl.position;
    const xyz = `xyz: (${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)})`;
    let rpy = 'rpy: (-, -, -)';
    if (ctrl.quaternion) {
      const q = new THREE.Quaternion(ctrl.quaternion.x, ctrl.quaternion.y, ctrl.quaternion.z, ctrl.quaternion.w);
      const e = new THREE.Euler().setFromQuaternion(q, 'XYZ');  // rb: roll=X, pitch=Y, yaw=Z
      const d = (v) => (v * 180 / Math.PI).toFixed(1);
      rpy = `rpy: (${d(e.x)}, ${d(e.y)}, ${d(e.z)})`;
    }
    return `${title}\n${xyz}\n${rpy}`;
  },

  _collectController: function (handEl, triggerDown, side) {
    const ctrl = {
      hand: side, position: null, quaternion: null,
      trigger: 0, gripActive: false,
      thumbstick: { x: 0, y: 0 }, buttons: {},
    };
    if (!handEl || !handEl.object3D) return ctrl;

    const pos_vr = handEl.object3D.position;
    const R_vr_g = handEl.object3D.quaternion;

    ctrl.trigger = triggerDown ? 1 : 0;
    const tracked = handEl.components && handEl.components['tracked-controls'];
    const gp = tracked && tracked.controller && tracked.controller.gamepad;
    if (gp) {
      const btn = (i) => gp.buttons[i] || { pressed: false, value: 0 };
      ctrl.trigger = btn(XR_BUTTONS.TRIGGER).value ?? (triggerDown ? 1 : 0);
      ctrl.thumbstick = { x: gp.axes[2] || 0, y: gp.axes[3] || 0 };
      ctrl.gripActive = btn(XR_BUTTONS.SQUEEZE).pressed;
      const common = {
        squeeze:    btn(XR_BUTTONS.SQUEEZE).pressed,
        thumbstick: btn(XR_BUTTONS.THUMBSTICK_CLICK).pressed,
      };
      ctrl.buttons = (side === 'left')
        ? { x: btn(XR_BUTTONS.PRIMARY).pressed, y: btn(XR_BUTTONS.SECONDARY).pressed, ...common }
        : { a: btn(XR_BUTTONS.PRIMARY).pressed, b: btn(XR_BUTTONS.SECONDARY).pressed, ...common };
    }

    // (vr, g) → (rb, ee)
    //   pos_rb  = R_rb_vr · pos_vr
    //   R_rb_ee = R_rb_vr · R_vr_g · R_g_ee     (R_g_ee = R_ee_g⁻¹)
    ctrl.position = { x: -pos_vr.z, y: -pos_vr.x, z: pos_vr.y };
    const R_rb_ee = new THREE.Quaternion()
      .multiplyQuaternions(this.R_rb_vr, R_vr_g)
      .multiply(this.R_ee_g[side].clone().conjugate());
    ctrl.quaternion = { x: R_rb_ee.x, y: R_rb_ee.y, z: R_rb_ee.z, w: R_rb_ee.w };
    return ctrl;
  },

  _collectHeadset: function () {
    const h = { position: null, quaternion: null };
    if (!this.headset || !this.headset.object3D) return h;
    const pos_vr = this.headset.object3D.position;
    h.position = { x: -pos_vr.z, y: -pos_vr.x, z: pos_vr.y };
    const R_rb_head = this._R_rb_head();
    h.quaternion = { x: R_rb_head.x, y: R_rb_head.y, z: R_rb_head.z, w: R_rb_head.w };
    return h;
  },
});


// ── Scene bootstrap ─────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  const scene = document.querySelector('a-scene');
  if (!scene) { console.error('a-scene not found'); return; }
  const addComp = () => scene.setAttribute('controller-updater', '');
  scene.hasLoaded ? addComp() : scene.addEventListener('loaded', addComp);

  if (navigator.xr) {
    navigator.xr.isSessionSupported('immersive-vr').then(ok => {
      if (ok) { addEnterVRButton(scene); return; }
      navigator.xr.isSessionSupported('immersive-ar').then(arOk => { if (arOk) addEnterVRButton(scene); });
    }).catch(err => console.warn('XR check failed:', err));
  }
});

function addEnterVRButton(scene) {
  const btn = document.createElement('button');
  btn.id = 'start-tracking-button';
  btn.textContent = 'Enter VR';
  Object.assign(btn.style, {
    position: 'fixed', top: '50%', left: '50%',
    transform: 'translate(-50%,-50%)',
    padding: '20px 40px', fontSize: '20px', fontWeight: 'bold',
    backgroundColor: '#4CAF50', color: 'white',
    border: 'none', borderRadius: '8px',
    cursor: 'pointer', zIndex: '9999',
    boxShadow: '0 4px 8px rgba(0,0,0,0.3)',
  });
  btn.onclick = () => scene.enterVR(true).catch(err => alert(`VR error: ${err.message}`));
  document.body.appendChild(btn);
  scene.addEventListener('enter-vr', () => { btn.style.display = 'none';  });
  scene.addEventListener('exit-vr',  () => { btn.style.display = 'block'; });
}
