/*
 * physics_anim.js -- 60 fps canvas animation of the physical system, driven ONLY by the PINN.
 *
 * The server sends (via the dcc.Store "anim-data") the network's prediction of each visualised
 * quantity over time, its autograd time-derivative ("rate"), the input signal acting on it
 * ("driver") and the reference solution (drawn as a dashed ghost). Playback, scrubbing and speed
 * are handled here so motion is smooth. Every moving thing is tied to a physical quantity:
 * positions/levels/colours come from the prediction, particle speeds and stream widths from the
 * predicted flows; when the physics is at rest the picture is at rest.
 */
(function () {
  "use strict";

  const C = {
    bg: "#0f1622", panel: "#111a27", line: "#2a3a50", grid: "#1c2635", text: "#d7e0ec", muted: "#8394ab",
    blue: "#4f9dff", orange: "#ff9f43", green: "#3ddc97", pink: "#ff6fae", ghost: "rgba(215,224,236,0.55)",
  };
  const THERMAL = [[0, [29, 78, 216]], [0.35, [34, 211, 238]], [0.6, [250, 204, 21]], [0.8, [251, 146, 60]], [1, [239, 68, 68]]];
  const SPEEDS = [0.25, 0.5, 1, 2, 4];
  const LOOP_SECONDS = 8; // full time span in 8 s of wall time at 1x

  // ---------------------------------------------------------------- helpers
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  function cmap(stops, u) {
    u = clamp(u, 0, 1);
    for (let i = 1; i < stops.length; i++) {
      if (u <= stops[i][0]) {
        const [p0, c0] = stops[i - 1], [p1, c1] = stops[i];
        const f = p1 === p0 ? 0 : (u - p0) / (p1 - p0);
        return c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
      }
    }
    return stops[stops.length - 1][1];
  }
  const rgba = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  function interp(ts, ys, t) {
    if (!ys || !ys.length) return null;
    const n = ts.length;
    if (t <= ts[0]) return ys[0];
    if (t >= ts[n - 1]) return ys[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ts[m] <= t) lo = m; else hi = m; }
    const a = ys[lo], b = ys[hi];
    if (a == null || b == null) return a ?? b;
    return a + (b - a) * (t - ts[lo]) / (ts[hi] - ts[lo]);
  }
  function fmt(v) {
    if (v == null || !isFinite(v)) return "–";
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(3);
    return (+v.toPrecision(4)).toString();
  }
  function rrect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function arrow(ctx, x0, y0, x1, y1, color, width) {
    const L = Math.hypot(x1 - x0, y1 - y0);
    if (L < 3) return;
    const ux = (x1 - x0) / L, uy = (y1 - y0) / L, h = Math.min(10, L * 0.45);
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - ux * h * 0.8, y1 - uy * h * 0.8); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - ux * h - uy * h * 0.55, y1 - uy * h + ux * h * 0.55);
    ctx.lineTo(x1 - ux * h + uy * h * 0.55, y1 - uy * h - ux * h * 0.55); ctx.closePath(); ctx.fill();
  }
  function label(ctx, text, x, y, opt = {}) {
    ctx.font = `${opt.weight || 400} ${opt.size || 11}px ${opt.mono ? "'JetBrains Mono', Consolas, monospace" : "Inter, 'Segoe UI', sans-serif"}`;
    ctx.fillStyle = opt.color || C.text; ctx.textAlign = opt.align || "center"; ctx.textBaseline = opt.base || "middle";
    ctx.fillText(text, x, y);
  }

  // --------------------------------------------------------------- widgets
  function drawThermal(ctx, b, w, s) {
    const T = s.v, Tr = s.ref, span = (w.hi - w.lo) || 1, u = (T - w.lo) / span;
    const Ts = s.drv;
    const hasSrc = Ts != null;
    const cy = b.y + b.h * 0.47, bh = Math.min(b.h * 0.5, 120);
    let bodyX, bodyW;
    if (hasSrc) {
      // heat reservoir at the source temperature
      const sx = b.x + b.w * 0.02, sw = b.w * 0.16;
      const cs = cmap(THERMAL, (Ts - w.lo) / span);
      rrect(ctx, sx, cy - bh * 0.62, sw, bh * 1.24, 8);
      ctx.fillStyle = rgba(cs, 0.9); ctx.fill(); ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.stroke();
      // heater coil (glow intensity = source temperature)
      ctx.strokeStyle = rgba([255, 240, 200], 0.35 + 0.6 * clamp((Ts - w.lo) / span, 0, 1)); ctx.lineWidth = 2;
      ctx.beginPath();
      for (let k = 0; k <= 24; k++) {
        const yy = cy - bh * 0.45 + (bh * 0.9) * k / 24, xx = sx + sw / 2 + Math.sin(k * Math.PI / 2) * sw * 0.22;
        k ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy);
      }
      ctx.stroke();
      label(ctx, `reservoir ${w.driver.name}`, sx + sw / 2, cy - bh * 0.62 - 12, { size: 10, color: C.muted });
      label(ctx, `${fmt(Ts)} ${w.driver.unit || ""}`, sx + sw / 2, cy + bh * 0.62 + 13, { size: 10, mono: true, color: C.text });

      // conduction rod (thermal resistance R): linear temperature profile between reservoir and body
      const rx0 = sx + sw, rx1 = b.x + b.w * 0.40, rh = bh * 0.26;
      const g = ctx.createLinearGradient(rx0, 0, rx1, 0);
      g.addColorStop(0, rgba(cs)); g.addColorStop(1, rgba(cmap(THERMAL, u)));
      ctx.fillStyle = g; ctx.fillRect(rx0, cy - rh / 2, rx1 - rx0, rh);
      ctx.strokeStyle = C.line; ctx.strokeRect(rx0, cy - rh / 2, rx1 - rx0, rh);
      label(ctx, "R (conduction)", (rx0 + rx1) / 2, cy + rh / 2 + 12, { size: 10, color: C.muted });
      // heat-flow particles: speed ∝ (T_src − T)  (flow through R is ΔT / R)
      const drive = clamp((Ts - T) / span, -1, 1);
      s.st.phase = (s.st.phase || 0) + s.dtw * drive * 0.9;
      const N = 12;
      for (let i = 0; i < N; i++) {
        let f = ((i / N + s.st.phase) % 1 + 1) % 1;
        const px = rx0 + f * (rx1 - rx0), py = cy + Math.sin(i * 2.4) * rh * 0.28;
        const a = Math.abs(drive) < 0.01 ? 0.15 : 0.95 * Math.sin(Math.PI * f);
        ctx.fillStyle = rgba([255, 214, 150], a);
        ctx.beginPath(); ctx.arc(px, py, 2.6, 0, 2 * Math.PI); ctx.fill();
      }
      bodyX = rx1; bodyW = b.w * 0.36;
    } else {
      bodyX = b.x + b.w * 0.14; bodyW = b.w * 0.56;
    }
    // the body, coloured by the PINN temperature
    const col = cmap(THERMAL, u);
    const glow = ctx.createRadialGradient(bodyX + bodyW / 2, cy, 4, bodyX + bodyW / 2, cy, bodyW * 0.75);
    glow.addColorStop(0, rgba(col, 0.35 * clamp(u, 0, 1) + 0.05)); glow.addColorStop(1, rgba(col, 0));
    ctx.fillStyle = glow; ctx.fillRect(bodyX - bodyW * 0.3, cy - bh, bodyW * 1.6, bh * 2);
    rrect(ctx, bodyX, cy - bh / 2, bodyW, bh, 10);
    const bg = ctx.createLinearGradient(0, cy - bh / 2, 0, cy + bh / 2);
    bg.addColorStop(0, rgba(col.map(v => Math.min(255, v + 35)))); bg.addColorStop(1, rgba(col.map(v => v * 0.75)));
    ctx.fillStyle = bg; ctx.fill(); ctx.strokeStyle = "rgba(0,0,0,0.5)"; ctx.lineWidth = 2; ctx.stroke();
    label(ctx, `${fmt(T)} ${w.unit}`, bodyX + bodyW / 2, cy, { size: 16, weight: 700, color: "#0b1018", mono: true });
    // heat shimmer above the body: amplitude grows with temperature
    const amp = 5 * clamp(u, 0, 1);
    if (amp > 0.3) {
      s.st.shim = (s.st.shim || 0) + s.dtw * (0.6 + 2.2 * clamp(u, 0, 1));
      ctx.strokeStyle = rgba(col, 0.45); ctx.lineWidth = 1.4;
      for (let k = 0; k < 3; k++) {
        const x0 = bodyX + bodyW * (0.25 + 0.25 * k);
        ctx.beginPath();
        for (let j = 0; j <= 20; j++) {
          const yy = cy - bh / 2 - 6 - j * 1.6, xx = x0 + Math.sin(j * 0.55 + s.st.shim * 4 + k) * amp;
          j ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy);
        }
        ctx.stroke();
      }
    }
    // net heat into the body (C·dT/dt from autograd): arrow in/out of the body
    const q = s.rate, qn = w.rate_max > 0 ? clamp(q / w.rate_max, -1, 1) : 0;
    const ax = bodyX + bodyW / 2, ay = cy + bh / 2 + 10;
    if (Math.abs(qn) > 0.02) {
      const L = 8 + 26 * Math.abs(qn);
      qn > 0 ? arrow(ctx, ax, ay + L, ax, ay, C.orange, 2.5) : arrow(ctx, ax, ay, ax, ay + L, C.blue, 2.5);
    }
    label(ctx, `net heat in = ${fmt(q)} ${w.rate_unit}`, ax, cy + bh / 2 + 48, { size: 10, mono: true, color: C.muted });
    // thermometer
    const tx = b.x + b.w * 0.86, tw = 12, ty0 = b.y + b.h * 0.12, ty1 = b.y + b.h * 0.8;
    rrect(ctx, tx - tw / 2, ty0, tw, ty1 - ty0, 6); ctx.fillStyle = "#0a111b"; ctx.fill(); ctx.strokeStyle = C.line; ctx.stroke();
    const hcol = ty0 + 4 + (ty1 - ty0 - 8) * (1 - clamp(u, 0, 1));
    ctx.fillStyle = rgba(col); ctx.fillRect(tx - tw / 2 + 3, hcol, tw - 6, ty1 - 4 - hcol);
    ctx.beginPath(); ctx.arc(tx, ty1 + 6, 9, 0, 2 * Math.PI); ctx.fillStyle = rgba(col); ctx.fill();
    if (Tr != null) {
      const yr = ty0 + 4 + (ty1 - ty0 - 8) * (1 - clamp((Tr - w.lo) / span, 0, 1));
      ctx.setLineDash([4, 3]); ctx.strokeStyle = C.ghost; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(tx - 14, yr); ctx.lineTo(tx + 14, yr); ctx.stroke(); ctx.setLineDash([]);
    }
    label(ctx, fmt(w.hi), tx + 16, ty0 + 4, { size: 9, color: C.muted, align: "left" });
    label(ctx, fmt(w.lo), tx + 16, ty1 - 4, { size: 9, color: C.muted, align: "left" });
  }

  function drawMassSpring(ctx, b, w, s) {
    const dmax = Math.max(Math.abs(w.lo), Math.abs(w.hi)) || 1;
    const ground = b.y + b.h * 0.78, wallX = b.x + b.w * 0.06, eq = b.x + b.w * 0.52, travel = b.w * 0.22;
    const mw = Math.min(b.w * 0.18, 90), mh = Math.min(b.h * 0.42, 90);
    const xc = eq + travel * clamp(s.v / dmax, -1.2, 1.2), left = xc - mw / 2;
    // ground + wall
    ctx.strokeStyle = C.line; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(wallX, ground); ctx.lineTo(b.x + b.w * 0.98, ground); ctx.stroke();
    ctx.lineWidth = 1; ctx.strokeStyle = "#26344a";
    for (let x = wallX + 6; x < b.x + b.w * 0.98; x += 12) { ctx.beginPath(); ctx.moveTo(x, ground); ctx.lineTo(x - 7, ground + 8); ctx.stroke(); }
    ctx.fillStyle = "#1b2636"; ctx.fillRect(wallX - 12, ground - mh * 1.4, 12, mh * 1.4);
    ctx.strokeStyle = "#33455e";
    for (let y = ground - mh * 1.4 + 4; y < ground; y += 10) { ctx.beginPath(); ctx.moveTo(wallX - 12, y + 7); ctx.lineTo(wallX, y); ctx.stroke(); }
    // motion trail (recent predicted positions)
    const trail = s.st.trail || (s.st.trail = []);
    trail.push(xc); if (trail.length > 14) trail.shift();
    trail.forEach((x, i) => { ctx.fillStyle = rgba([255, 159, 67], 0.03 * i); ctx.fillRect(x - mw / 2, ground - mh, mw, mh); });
    // spring (fixed number of coils -> visibly stretches/compresses)
    const ys = ground - mh * 0.7, coils = 9, amp = mh * 0.13;
    ctx.strokeStyle = C.blue; ctx.lineWidth = 2.2; ctx.beginPath(); ctx.moveTo(wallX, ys);
    const x0 = wallX + 10, x1 = left - 10;
    ctx.lineTo(x0, ys);
    for (let k = 0; k <= coils * 2; k++) ctx.lineTo(x0 + (x1 - x0) * k / (coils * 2), ys + (k === 0 || k === coils * 2 ? 0 : (k % 2 ? -amp : amp)));
    ctx.lineTo(left, ys); ctx.stroke();
    // damper: cylinder fixed to the wall, piston rod fixed to the mass
    const yd = ground - mh * 0.28, cylEnd = wallX + (eq - mw / 2 - wallX) * 0.62;
    ctx.strokeStyle = C.muted; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(wallX, yd); ctx.lineTo(wallX + 14, yd); ctx.stroke();
    ctx.strokeRect(wallX + 14, yd - 9, cylEnd - wallX - 14, 18);
    const xp = clamp(wallX + 14 + (left - wallX - 14) * 0.55, wallX + 18, cylEnd - 4);
    ctx.strokeStyle = C.orange; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(xp, yd - 7); ctx.lineTo(xp, yd + 7); ctx.stroke();
    ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xp, yd); ctx.lineTo(left, yd); ctx.stroke();
    // reference ghost
    if (s.ref != null) {
      const xr = eq + travel * clamp(s.ref / dmax, -1.2, 1.2);
      ctx.setLineDash([5, 4]); ctx.strokeStyle = C.ghost; ctx.lineWidth = 1.2;
      ctx.strokeRect(xr - mw / 2, ground - mh, mw, mh); ctx.setLineDash([]);
    }
    // mass
    rrect(ctx, left, ground - mh, mw, mh, 6);
    const mg = ctx.createLinearGradient(0, ground - mh, 0, ground);
    mg.addColorStop(0, "#35507a"); mg.addColorStop(1, "#22324c");
    ctx.fillStyle = mg; ctx.fill(); ctx.strokeStyle = C.orange; ctx.lineWidth = 2; ctx.stroke();
    label(ctx, "m", xc, ground - mh / 2, { size: 16, weight: 700 });
    // equilibrium marker
    ctx.setLineDash([2, 3]); ctx.strokeStyle = C.muted; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(eq, ground - mh - 26); ctx.lineTo(eq, ground + 2); ctx.stroke(); ctx.setLineDash([]);
    label(ctx, "x = 0", eq, ground - mh - 32, { size: 9, color: C.muted });
    // velocity arrow (dx/dt from autograd) above the mass
    const vn = w.rate_max > 0 ? clamp(s.rate / w.rate_max, -1, 1) : 0;
    arrow(ctx, xc, ground - mh - 12, xc + vn * mw * 0.9, ground - mh - 12, C.green, 2.5);
    // force arrow from the input signal
    if (s.drv != null && w.driver.max > 0) {
      const fn = clamp(s.drv / w.driver.max, -1, 1);
      arrow(ctx, left + mw + 4, ground - mh / 2, left + mw + 4 + fn * 46, ground - mh / 2, C.pink, 3);
      label(ctx, `${w.driver.name} = ${fmt(s.drv)} ${w.driver.unit}`, left + mw + 8, ground - mh / 2 - 14, { size: 10, color: C.pink, align: "left", mono: true });
    }
    label(ctx, `x = ${fmt(s.v)} ${w.unit}   v = ${fmt(s.rate)} ${w.rate_unit}`, b.x + b.w / 2, b.y + 12, { size: 10, mono: true, color: C.muted });
    label(ctx, `motion magnified · full travel = ±${fmt(dmax)} ${w.unit}`, b.x + b.w / 2, ground + 18, { size: 9, color: C.muted });
  }

  function drawRotor(ctx, b, w, s) {
    const cx = b.x + b.w * 0.5, cy = b.y + b.h * 0.47, r = Math.min(b.w, b.h) * 0.3;
    const wmax = Math.max(Math.abs(w.lo), Math.abs(w.hi)) || 1, wn = clamp(Math.abs(s.v) / wmax, 0, 1);
    const theta = s.angle || 0;
    // shaft + bearing
    ctx.fillStyle = "#1b2636"; ctx.fillRect(cx - r - 26, cy - 5, 2 * r + 52, 10);
    const disc = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.3, 2, cx, cy, r);
    const dc = cmap([[0, [31, 58, 95]], [1, [184, 98, 27]]], wn);
    disc.addColorStop(0, rgba(dc.map(v => Math.min(255, v + 40)))); disc.addColorStop(1, rgba(dc));
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.fillStyle = disc; ctx.fill();
    ctx.strokeStyle = C.orange; ctx.lineWidth = 2; ctx.stroke();
    for (let k = 0; k < 6; k++) {
      const a = -theta + k * Math.PI / 3;
      ctx.strokeStyle = "rgba(230,237,247,0.85)"; ctx.lineWidth = k === 0 ? 4 : 2;
      ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * r * 0.16, cy + Math.sin(a) * r * 0.16);
      ctx.lineTo(cx + Math.cos(a) * r * 0.9, cy + Math.sin(a) * r * 0.9); ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(cx + Math.cos(-theta) * r * 0.9, cy + Math.sin(-theta) * r * 0.9, 5, 0, 2 * Math.PI);
    ctx.fillStyle = C.orange; ctx.fill();
    ctx.beginPath(); ctx.arc(cx, cy, r * 0.14, 0, 2 * Math.PI); ctx.fillStyle = C.bg; ctx.fill(); ctx.strokeStyle = C.text; ctx.stroke();
    // speed lines trailing the marker just outside the rim (arc length ∝ angular speed)
    const dir = Math.sign(s.v || 1);
    for (let k = 0; k < 3; k++) {
      ctx.strokeStyle = rgba([255, 159, 67], (0.5 - 0.13 * k) * wn); ctx.lineWidth = 2.5 - 0.6 * k;
      ctx.beginPath(); ctx.arc(cx, cy, r * (1.08 + 0.07 * k), -theta, -theta + dir * wn * 1.6, dir < 0); ctx.stroke();
    }
    label(ctx, `ω = ${fmt(s.v)} ${w.unit}`, cx, cy + r + 18, { size: 12, mono: true, weight: 600 });
    label(ctx, `angle = ∫ω dt = ${theta.toFixed(1)} rad` + (s.ref != null ? `   ·   ref ω ${fmt(s.ref)}` : ""), cx, cy + r + 34, { size: 9, color: C.muted, mono: true });
    if (s.drv != null) label(ctx, `${w.driver.name} = ${fmt(s.drv)} ${w.driver.unit}`, cx, b.y + 12, { size: 10, color: C.pink, mono: true });
  }

  function drawTank(ctx, b, w, s) {
    const base = Math.min(w.lo, 0), span = (w.hi - base) || 1;
    const tx = b.x + b.w * 0.28, tw = b.w * 0.44, ty = b.y + b.h * 0.14, th = b.h * 0.66;
    const lvl = clamp((s.v - base) / span, 0, 1), yl = ty + th - th * lvl;
    const qn = w.rate_max > 0 ? clamp(s.rate / w.rate_max, -1, 1) : 0;
    s.st.wave = (s.st.wave || 0) + s.dtw * (1 + 5 * Math.abs(qn));
    // inflow pipe + stream (width ∝ predicted net inflow)
    ctx.fillStyle = "#1b2636"; ctx.fillRect(b.x + b.w * 0.06, ty - 8, tx - b.x - b.w * 0.06 + tw * 0.25, 12);
    if (qn > 0.01) {
      const sw = 2 + 12 * qn, sx = tx + tw * 0.25 - sw / 2;
      ctx.fillStyle = "rgba(79,157,255,0.7)"; ctx.fillRect(sx, ty + 4, sw, yl - ty - 4);
      for (let k = 0; k < 6; k++) {
        const f = ((k / 6 + s.st.wave * 0.25) % 1), yy = ty + 4 + f * (yl - ty - 4);
        ctx.fillStyle = "rgba(200,230,255,0.8)"; ctx.fillRect(sx + sw * 0.3, yy, sw * 0.4, 4);
      }
    }
    // outflow (when the level is falling)
    ctx.fillStyle = "#1b2636"; ctx.fillRect(tx + tw, ty + th - 14, b.w * 0.14, 10);
    if (qn < -0.01) {
      const sw = 2 + 10 * -qn;
      ctx.fillStyle = "rgba(79,157,255,0.7)"; ctx.fillRect(tx + tw + b.w * 0.14 - sw, ty + th - 4, sw, b.h * 0.12);
    }
    // water with a moving surface (amplitude ∝ |net flow|)
    const A = 1 + 5 * Math.abs(qn);
    ctx.beginPath(); ctx.moveTo(tx + 2, ty + th - 2);
    for (let k = 0; k <= 30; k++) { const xx = tx + 2 + (tw - 4) * k / 30; ctx.lineTo(xx, yl + Math.sin(k * 0.7 - s.st.wave * 3) * A); }
    ctx.lineTo(tx + tw - 2, ty + th - 2); ctx.closePath();
    const wg = ctx.createLinearGradient(0, yl, 0, ty + th); wg.addColorStop(0, "rgba(79,157,255,0.85)"); wg.addColorStop(1, "rgba(29,78,160,0.9)");
    ctx.fillStyle = wg; ctx.fill();
    // tank walls
    ctx.strokeStyle = C.muted; ctx.lineWidth = 2.5; ctx.beginPath();
    ctx.moveTo(tx, ty); ctx.lineTo(tx, ty + th); ctx.lineTo(tx + tw, ty + th); ctx.lineTo(tx + tw, ty); ctx.stroke();
    if (s.ref != null) {
      const yr = ty + th - th * clamp((s.ref - base) / span, 0, 1);
      ctx.setLineDash([5, 4]); ctx.strokeStyle = C.ghost; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(tx - 10, yr); ctx.lineTo(tx + tw + 10, yr); ctx.stroke(); ctx.setLineDash([]);
    }
    label(ctx, `${fmt(s.v)} ${w.unit}`, tx + tw / 2, ty + th + 16, { size: 12, mono: true, weight: 600 });
    label(ctx, `net inflow = ${fmt(s.rate)} ${w.rate_unit}`, tx + tw / 2, ty + th + 31, { size: 9, mono: true, color: C.muted });
  }

  function drawGauge(ctx, b, w, s) {
    const lo = Math.min(w.lo, 0), hi = Math.max(w.hi, 0), span = (hi - lo) || 1;
    const cx = b.x + b.w / 2, cy = b.y + b.h * 0.66, r = Math.min(b.w * 0.42, b.h * 0.55);
    const ang = v => Math.PI + Math.PI * clamp((v - lo) / span, 0, 1);
    ctx.lineWidth = 10; ctx.strokeStyle = "#1b2636";
    ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI); ctx.stroke();
    ctx.strokeStyle = (s.v ?? 0) >= 0 ? C.orange : C.blue;
    ctx.beginPath(); ctx.arc(cx, cy, r, Math.min(ang(0), ang(s.v)), Math.max(ang(0), ang(s.v))); ctx.stroke();
    ctx.lineWidth = 1; ctx.strokeStyle = C.line;
    for (let k = 0; k <= 10; k++) {
      const a = Math.PI + Math.PI * k / 10;
      ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * (r - 14), cy + Math.sin(a) * (r - 14)); ctx.lineTo(cx + Math.cos(a) * (r - 6), cy + Math.sin(a) * (r - 6)); ctx.stroke();
    }
    if (s.ref != null) {
      ctx.setLineDash([4, 3]); ctx.strokeStyle = C.ghost; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(ang(s.ref)) * (r - 4), cy + Math.sin(ang(s.ref)) * (r - 4)); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.strokeStyle = C.text; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(ang(s.v)) * (r - 10), cy + Math.sin(ang(s.v)) * (r - 10)); ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, cy, 5, 0, 2 * Math.PI); ctx.fillStyle = C.text; ctx.fill();
    label(ctx, fmt(lo), cx - r, cy + 14, { size: 9, color: C.muted });
    label(ctx, fmt(hi), cx + r, cy + 14, { size: 9, color: C.muted });
    label(ctx, `${fmt(s.v)} ${w.unit}`, cx, cy + 22, { size: 12, mono: true, weight: 600 });
  }

  const DRAW = { thermal: drawThermal, mass_spring: drawMassSpring, rotor: drawRotor, tank: drawTank, gauge: drawGauge };

  // mini time-history under each widget
  function drawTrace(ctx, b, w, data, t) {
    ctx.fillStyle = "#0b121c"; rrect(ctx, b.x, b.y, b.w, b.h, 6); ctx.fill();
    const ts = data.t, lo = w.lo, hi = w.hi, span = (hi - lo) || 1;
    const X = tt => b.x + 6 + (b.w - 12) * (tt - data.t0) / (data.t1 - data.t0);
    const Y = v => b.y + b.h - 6 - (b.h - 12) * (v - lo) / span;
    if (w.ref) {
      ctx.setLineDash([4, 3]); ctx.strokeStyle = C.ghost; ctx.lineWidth = 1; ctx.beginPath();
      w.ref.forEach((v, i) => { if (v != null) (i ? ctx.lineTo(X(ts[i]), Y(v)) : ctx.moveTo(X(ts[i]), Y(v))); });
      ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.lineWidth = 2; ctx.strokeStyle = C.orange; ctx.beginPath();
    let started = false;
    for (let i = 0; i < ts.length && ts[i] <= t; i++) {
      const v = w.pred[i]; if (v == null) continue;
      started ? ctx.lineTo(X(ts[i]), Y(v)) : (ctx.moveTo(X(ts[i]), Y(v)), started = true);
    }
    ctx.stroke();
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(255,159,67,0.25)"; ctx.beginPath(); started = false;
    for (let i = 0; i < ts.length; i++) {
      if (ts[i] < t) continue; const v = w.pred[i]; if (v == null) continue;
      started ? ctx.lineTo(X(ts[i]), Y(v)) : (ctx.moveTo(X(ts[i]), Y(v)), started = true);
    }
    ctx.stroke();
    const v = interp(ts, w.pred, t);
    ctx.strokeStyle = "rgba(61,220,151,0.5)"; ctx.beginPath(); ctx.moveTo(X(t), b.y + 3); ctx.lineTo(X(t), b.y + b.h - 3); ctx.stroke();
    if (v != null) { ctx.fillStyle = C.green; ctx.beginPath(); ctx.arc(X(t), Y(v), 3.5, 0, 2 * Math.PI); ctx.fill(); }
  }

  // ---------------------------------------------------------------- engine
  class Anim {
    constructor(root) {
      this.root = root; this.data = null; this.t = null; this.playing = true; this.speed = 1;
      this.state = {}; this.last = null;
      root.innerHTML = "";
      this.canvas = document.createElement("canvas"); this.canvas.className = "anim-canvas";
      this.msg = document.createElement("div"); this.msg.className = "anim-msg";
      const bar = document.createElement("div"); bar.className = "anim-bar";
      this.btn = document.createElement("button"); this.btn.className = "btn icon"; this.btn.textContent = "❚❚";
      this.btn.title = "Play / pause";
      this.btn.onclick = () => { this.playing = !this.playing; this.btn.textContent = this.playing ? "❚❚" : "▶"; };
      this.range = document.createElement("input"); this.range.type = "range"; this.range.min = 0; this.range.max = 1000;
      this.range.value = 0; this.range.className = "anim-range";
      this.range.oninput = () => {
        if (!this.data) return;
        this.t = this.data.t0 + (this.data.t1 - this.data.t0) * this.range.value / 1000; this.scrubbing = true;
      };
      this.range.onchange = () => { this.scrubbing = false; };
      this.sel = document.createElement("select"); this.sel.className = "anim-speed"; this.sel.title = "Playback speed";
      SPEEDS.forEach(v => { const o = document.createElement("option"); o.value = v; o.textContent = `${v}×`; if (v === 1) o.selected = true; this.sel.appendChild(o); });
      this.sel.onchange = () => { this.speed = +this.sel.value; };
      this.tlabel = document.createElement("span"); this.tlabel.className = "anim-time mono";
      bar.append(this.btn, this.range, this.sel, this.tlabel);
      root.append(this.canvas, this.msg, bar);
      this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(root);
      this.resize();
      requestAnimationFrame(ts => this.frame(ts));
    }
    resize() {
      const dpr = window.devicePixelRatio || 1, w = this.root.clientWidth || 600, h = 360;
      this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
      this.canvas.style.width = w + "px"; this.canvas.style.height = h + "px";
      this.W = w; this.H = h; this.dpr = dpr;
    }
    update(d) {
      this.data = d && d.ready ? d : null;
      this.msg.textContent = d && !d.ready ? (d.message || "") : "";
      this.msg.style.display = this.data ? "none" : "flex";
      if (!this.data) return;
      const key = this.data.widgets.map(w => w.kind + w.obs).join("|") + this.data.t0 + this.data.t1;
      if (key !== this.key) { this.key = key; this.state = {}; this.t = this.data.t0; }
      // rotor angle = cumulative trapezoid of the predicted angular velocity
      this.data.widgets.forEach(w => {
        if (w.kind !== "rotor") return;
        const ts = this.data.t, a = [0];
        for (let i = 1; i < ts.length; i++) a.push(a[i - 1] + 0.5 * ((w.pred[i] || 0) + (w.pred[i - 1] || 0)) * (ts[i] - ts[i - 1]));
        w.angle = a;
      });
    }
    frame(now) {
      const dtw = this.last == null ? 0 : Math.min(0.1, (now - this.last) / 1000);
      this.last = now;
      if (this.data && this.playing && !this.scrubbing) {
        const span = this.data.t1 - this.data.t0;
        this.t += dtw * span * this.speed / LOOP_SECONDS;
        if (this.t > this.data.t1) this.t = this.data.t0;
        this.range.value = Math.round(1000 * (this.t - this.data.t0) / span);
      }
      try { this.draw(this.playing ? dtw * this.speed : 0); } catch (e) { /* keep the loop alive */ console.error(e); }
      requestAnimationFrame(ts => this.frame(ts));
    }
    draw(dtw) {
      const ctx = this.canvas.getContext("2d");
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.clearRect(0, 0, this.W, this.H);
      const d = this.data;
      if (!d) { this.tlabel.textContent = ""; return; }
      const t = this.t, n = d.widgets.length, gap = 12, pw = (this.W - gap * (n - 1)) / n;
      const traceH = 62, titleH = 34;
      d.widgets.forEach((w, i) => {
        const x = i * (pw + gap);
        ctx.fillStyle = C.panel; rrect(ctx, x, 0, pw, this.H, 10); ctx.fill();
        label(ctx, w.title, x + pw / 2, 13, { size: 12, weight: 700 });
        label(ctx, w.label, x + pw / 2, 27, { size: 10, color: C.muted });
        const st = this.state[i] || (this.state[i] = {});
        const s = {
          v: interp(d.t, w.pred, t), ref: w.ref ? interp(d.t, w.ref, t) : null,
          rate: interp(d.t, w.rate, t), drv: w.driver ? interp(d.t, w.driver.values, t) : null,
          angle: w.angle ? interp(d.t, w.angle, t) : null, st, dtw,
        };
        const box = { x: x + 8, y: titleH, w: pw - 16, h: this.H - titleH - traceH - 12 };
        DRAW[w.kind](ctx, box, w, s);
        drawTrace(ctx, { x: x + 8, y: this.H - traceH - 6, w: pw - 16, h: traceH }, w, d, t);
      });
      this.tlabel.textContent = `t = ${fmt(t)}  ·  PINN epoch ${d.epoch}` + (d.has_ref ? "  ·  dashed = reference" : "");
    }
  }

  window.PINNAnim = {
    update(data) {
      const root = document.getElementById("anim-root");
      if (!root) return;
      if (!this.inst || this.inst.root !== root) this.inst = new Anim(root);
      this.inst.update(data);
    },
  };
})();
