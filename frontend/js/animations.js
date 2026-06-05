/* ═══════════════════════════════════════════════════
   FERROMIND — animations.js  (FULL PREMIUM UPGRADE)
   ═══════════════════════════════════════════════════
   FIXES vs previous version
   ──────────────────────────────────────────────────
   [A] reveal / reveal-left / reveal-right classes now
       injected onto elements + observed by IntersectionObserver
   [B] kpi-val.ticking class toggled on counter finish
   [C] fab.pulse re-added when drawer closes
   [D] staggerRows also fires for metrics grade table
   [E] cursor grow wired to dynamically created elements
       (opt-cells, model-cards, grade-pills) via delegation
   [F] kpi::before width animation — JS forces width:0
       before observer fires so the CSS transition plays
   ═══════════════════════════════════════════════════ */
(function () {
'use strict';

/* ══ 1. LOADING SCREEN ══ */
function initLoader() {
  const el = document.getElementById('loadScreen');
  if (!el) return;
  let p = 0;
  const pct = document.getElementById('loadPercent');
  const iv = setInterval(() => {
    p = Math.min(p + Math.random() * 18, 99);
    if (pct) pct.textContent = Math.round(p) + '%';
  }, 120);
  setTimeout(() => {
    clearInterval(iv);
    if (pct) pct.textContent = '100%';
    setTimeout(() => {
      el.classList.add('fade-out');
      setTimeout(() => el.remove(), 800);
    }, 300);
  }, 1500);
}

/* ══ 2. CUSTOM CURSOR ══ */
function initCursor() {
  // skip on touch-only devices
  if (window.matchMedia('(pointer: coarse)').matches) return;

  const dot  = document.createElement('div'); dot.id  = 'cursor-dot';
  const ring = document.createElement('div'); ring.id = 'cursor-ring';
  document.body.appendChild(dot);
  document.body.appendChild(ring);

  let mx = 0, my = 0, rx = 0, ry = 0;

  document.addEventListener('mousemove', e => {
    mx = e.clientX; my = e.clientY;
    dot.style.left = mx + 'px';
    dot.style.top  = my + 'px';
  });

  // ring lags behind (smooth follow)
  function followRing() {
    rx += (mx - rx) * 0.12;
    ry += (my - ry) * 0.12;
    ring.style.left = rx + 'px';
    ring.style.top  = ry + 'px';
    requestAnimationFrame(followRing);
  }
  followRing();

  // [E] Delegation — covers static + dynamically injected elements
  const INTERACTIVE = 'button, a, input, select, .h-tab, .kpi, .card, ' +
                      '.comp-cell, .opt-cell, .model-card, .grade-pill, .a-kpi';

  document.addEventListener('mouseover', e => {
    if (e.target.closest(INTERACTIVE)) {
      dot.style.width  = '14px';
      dot.style.height = '14px';
      dot.style.background = 'var(--gold)';
    }
  });
  document.addEventListener('mouseout', e => {
    if (e.target.closest(INTERACTIVE)) {
      dot.style.width  = '8px';
      dot.style.height = '8px';
    }
  });
}

/* ══ 3. FLOATING PARTICLES ══ */
function initParticles() {
  const canvas = document.createElement('canvas');
  canvas.id = 'particleCanvas';
  document.body.insertBefore(canvas, document.body.firstChild);
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  const GOLD = 'rgba(232,160,32,';
  const BLUE = 'rgba(30,111,191,';
  const N = 48;

  const pts = Array.from({ length: N }, () => ({
    x:  Math.random() * window.innerWidth,
    y:  Math.random() * window.innerHeight,
    r:  Math.random() * 1.8 + 0.4,
    vx: (Math.random() - .5) * .35,
    vy: (Math.random() - .5) * .35,
    c:  Math.random() > .5 ? GOLD : BLUE,
    o:  Math.random() * .5 + .2,
  }));

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = pts[i].x - pts[j].x;
        const dy = pts[i].y - pts[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 130) {
          ctx.beginPath();
          ctx.moveTo(pts[i].x, pts[i].y);
          ctx.lineTo(pts[j].x, pts[j].y);
          ctx.strokeStyle = `rgba(232,160,32,${(1 - dist/130) * .08})`;
          ctx.lineWidth = .6;
          ctx.stroke();
        }
      }
    }

    pts.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.c + p.o + ')';
      ctx.fill();
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > canvas.width)  p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
    });

    requestAnimationFrame(draw);
  }
  draw();
}

/* ══ 4. SCROLL HEADER HIDE/SHOW ══ */
function initScrollHeader() {
  let lastY = 0;
  window.addEventListener('scroll', () => {
    const y   = window.scrollY;
    const hdr = document.querySelector('header');
    if (!hdr) return;
    hdr.classList.toggle('scrolled', y > 10);
    hdr.classList.toggle('hidden-up', y > lastY && y > 100);
    lastY = y;
  }, { passive: true });
}

/* ══ 5. INTERSECTION OBSERVER — scroll reveal ══
   [A] Observes .reveal / .reveal-left / .reveal-right (injected by
       injectRevealClasses) AND all existing card/kpi/etc. selectors.
   [F] Resets kpi::before width so CSS transition re-fires.           */
function initScrollReveal() {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      e.target.classList.add('visible', 'in-view');
      obs.unobserve(e.target);
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

  function observe() {
    // [F] Force kpi::before back to 0 width before re-observing
    document.querySelectorAll('.kpi:not(.in-view)').forEach(k => {
      k.style.setProperty('--bar-w', '0');
    });

    document.querySelectorAll(
      '.card, .section-head, .a-kpi, .insight, .kpi, ' +
      '.model-card, .comp-cell, .opt-cell, .opt-saving, ' +
      '.reveal, .reveal-left, .reveal-right'           // [A]
    ).forEach(el => {
      if (!el.classList.contains('visible')) obs.observe(el);
    });
  }
  observe();

  document.querySelectorAll('.h-tab').forEach(t => {
    t.addEventListener('click', () => setTimeout(observe, 60));
  });
}

/* ══ 5b. INJECT REVEAL CLASSES onto static elements ══
   [A] Cards in grid-2/grid-3 get alternating reveal-left / reveal-right.
   Section heads get reveal. Insight boxes get reveal.                 */
function injectRevealClasses() {
  // section heads
  document.querySelectorAll('.section-head').forEach(el => {
    el.classList.add('reveal-left');
  });

  // insight boxes
  document.querySelectorAll('.insight').forEach(el => {
    el.classList.add('reveal');
  });

  // cards inside two-column grids → alternate left/right
  document.querySelectorAll('.grid-2').forEach(grid => {
    const cards = grid.querySelectorAll('.card');
    cards.forEach((c, i) => {
      c.classList.add(i % 2 === 0 ? 'reveal-left' : 'reveal-right');
    });
  });

  // cards inside three-column grids → stagger reveal
  document.querySelectorAll('.grid-3').forEach(grid => {
    grid.querySelectorAll('.card').forEach(c => c.classList.add('reveal'));
  });

  // full-width cards → simple reveal
  document.querySelectorAll('.grid-full > .card').forEach(c => {
    if (!c.classList.contains('reveal-left') && !c.classList.contains('reveal-right'))
      c.classList.add('reveal');
  });
}

/* ══ 6. KPI NUMBER COUNTER ══
   [B] Adds .ticking to the element while animating, removes on finish. */
function countUp(el, target, decimals, duration, delay) {
  setTimeout(() => {
    el.classList.add('ticking');              // [B] triggers numTick CSS anim
    const start = performance.now();
    function step(now) {
      const p    = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(2, -10 * p);
      const val  = target * ease;
      el.textContent = decimals > 0
        ? val.toFixed(decimals)
        : Math.round(val).toLocaleString();
      if (p < 1) {
        requestAnimationFrame(step);
      } else {
        el.classList.remove('ticking');      // [B] clean up after done
      }
    }
    requestAnimationFrame(step);
  }, delay);
}

function runCounters() {
  const heats = document.getElementById('kpiHeats');
  if (heats) countUp(heats, 47421, 0, 1600, 400);

  const mnR2 = document.getElementById('kpiMnR2');
  if (mnR2) { const v = parseFloat(mnR2.textContent)||.965; mnR2.textContent='0.000'; countUp(mnR2, v, 3, 1400, 550); }

  const cR2 = document.getElementById('kpiCR2');
  if (cR2)  { const v = parseFloat(cR2.textContent)||.922; cR2.textContent='0.000'; countUp(cR2,  v, 3, 1400, 700); }
}

/* ══ 7. BUTTON RIPPLE ══ */
function addRipple(btn) {
  btn.addEventListener('click', e => {
    const r = btn.getBoundingClientRect();
    const s = document.createElement('span');
    s.style.cssText =
      `position:absolute;left:${e.clientX - r.left}px;top:${e.clientY - r.top}px;` +
      `width:0;height:0;border-radius:50%;pointer-events:none;` +
      `background:rgba(255,255,255,.22);transform:translate(-50%,-50%);` +
      `animation:rippleOut .55s cubic-bezier(.19,1,.22,1) forwards;`;
    btn.appendChild(s);
    setTimeout(() => s.remove(), 600);
  });
}

/* ══ 8. DRAWER ANIMATIONS ══
   [C] Re-adds .pulse to FAB when drawer closes.                       */
function replayDrawer() {
  const drawer = document.getElementById('drawer');
  if (!drawer) return;
  drawer.querySelectorAll(
    '.d-field,.drawer-title,.drawer-close,.drawer-section-label,.btn-drawer-predict,.btn-drawer-opt'
  ).forEach(el => {
    el.style.animation = 'none';
    void el.offsetWidth;   // force reflow
    el.style.animation = '';
  });
}

function patchDrawer() {
  const oOpen  = window.openDrawer;
  const oClose = window.closeDrawer;

  window.openDrawer = function () {
    if (typeof oOpen === 'function') oOpen();
    // remove pulse while open
    const fab = document.getElementById('fab');
    if (fab) fab.classList.remove('pulse');
    requestAnimationFrame(() => requestAnimationFrame(replayDrawer));
  };

  window.closeDrawer = function () {
    if (typeof oClose === 'function') oClose();
    // [C] restore pulse when drawer is dismissed
    const fab = document.getElementById('fab');
    if (fab) {
      setTimeout(() => fab.classList.add('pulse'), 400);
    }
  };
}

/* ══ 9. TAB SWITCH — re-trigger observers + counters ══ */
function patchSwitchTab() {
  const orig = window.switchTab;
  if (typeof orig !== 'function') return;

  window.switchTab = function (name) {
    orig(name);
    setTimeout(() => {
      injectRevealClasses();   // inject onto any newly-visible elements
      initScrollReveal();      // re-observe

      if (name === 'analytics') {
        document.querySelectorAll('.a-kpi-val').forEach(el => {
          const n   = parseFloat(el.textContent);
          if (!isNaN(n) && n > 0) {
            const dec = el.textContent.includes('.') ? 3 : 0;
            el.textContent = dec ? '0.000' : '0';
            countUp(el, n, dec, 1000, 200);
          }
        });
      }

      // [D] stagger rows in metrics grade table when that tab opens
      if (name === 'metrics') {
        setTimeout(() => staggerRows('gradeTable'), 200);
      }
    }, 60);
  };
}

/* ══ 10. TABLE ROW STAGGER ══
   [D] Accepts a container ID — finds all tbody tr inside it.         */
function staggerRows(containerId) {
  const rows = document.querySelectorAll(`#${containerId} tbody tr`);
  rows.forEach((r, i) => {
    r.style.animationDelay     = (i * 35) + 'ms';
    r.style.animationPlayState = 'running';
  });
}

// patch loadRecent — stagger recent heats table
const _origRecent = window.loadRecent;
if (typeof _origRecent === 'function') {
  window.loadRecent = async function () {
    await _origRecent();
    setTimeout(() => staggerRows('recentTable'), 100);
  };
}

/* ══ 11. PREDICT — re-animate cells ══ */
const _origPredict = window.runPredict;
if (typeof _origPredict === 'function') {
  window.runPredict = async function () {
    document.querySelectorAll('.comp-cell').forEach(c => {
      c.style.animation = 'none';
      void c.offsetWidth;
      c.style.animation = '';
    });
    await _origPredict();
  };
}

/* ══ 12. OPTIMISE — re-animate opt-cells after result renders ══ */
const _origOptimise = window.runOptimise;
if (typeof _origOptimise === 'function') {
  window.runOptimise = async function () {
    await _origOptimise();
    // After DOM update, stagger new opt-cells
    setTimeout(() => {
      document.querySelectorAll('#optResult .opt-cell').forEach((c, i) => {
        c.style.animationDelay = (i * 70) + 'ms';
      });
      document.querySelectorAll('#optResult .opt-saving').forEach(c => {
        c.style.animationDelay = '0.3s';
      });
      // also wire cursor grow onto fresh cells
      document.querySelectorAll('#optResult .comp-cell').forEach(c => {
        // handled by delegation in initCursor, nothing extra needed
      });
    }, 80);
  };
}

/* ══ 12b. RIPPLE on all action buttons ══ */
function initRipples() {
  document.querySelectorAll(
    '.btn-predict,.btn-optimise,.btn-drawer-predict,.btn-drawer-opt'
  ).forEach(addRipple);
}

/* ══ 13. TOAST NOTIFICATION SYSTEM ══ */
let _toastId = 0;
const MAX_TOASTS = 5;

function _ensureContainer() {
  let c = document.getElementById('toastContainer');
  if (!c) {
    c = document.createElement('div');
    c.id = 'toastContainer';
    document.body.appendChild(c);
  }
  return c;
}

const TOAST_ICONS = {
  success: '✓',
  error:   '✕',
  info:    'ℹ',
  loading: '⟳',
};
const TOAST_TITLES = {
  success: 'Success',
  error:   'Error',
  info:    'Info',
  loading: 'Processing',
};

window.showToast = function(message, type, duration) {
  type = type || 'info';
  duration = duration || (type === 'loading' ? 0 : 4000);
  const container = _ensureContainer();
  const id = 'toast-' + (++_toastId);

  // Limit visible toasts
  const existing = container.querySelectorAll('.toast:not(.exiting)');
  if (existing.length >= MAX_TOASTS) {
    const oldest = existing[existing.length - 1];
    _dismissToast(oldest);
  }

  const toast = document.createElement('div');
  toast.className = 'toast ' + type;
  toast.id = id;
  if (duration > 0) toast.style.setProperty('--toast-dur', duration + 'ms');

  toast.innerHTML =
    '<div class="toast-icon">' + TOAST_ICONS[type] + '</div>' +
    '<div class="toast-body">' +
      '<div class="toast-title">' + TOAST_TITLES[type] + '</div>' +
      '<div class="toast-msg">' + message + '</div>' +
    '</div>' +
    '<button class="toast-close" onclick="dismissToast(\'' + id + '\')">&times;</button>' +
    (duration > 0 ? '<div class="toast-progress"></div>' : (type === 'loading' ? '<div class="toast-progress"></div>' : ''));

  container.insertBefore(toast, container.firstChild);

  if (duration > 0) {
    setTimeout(function() { _dismissToast(toast); }, duration);
  }

  return id;
};

function _dismissToast(el) {
  if (!el || el.classList.contains('exiting')) return;
  el.classList.add('exiting');
  setTimeout(function() { if (el.parentNode) el.parentNode.removeChild(el); }, 450);
}

window.dismissToast = function(id) {
  const el = document.getElementById(id);
  if (el) _dismissToast(el);
};

window.updateToast = function(id, message, type) {
  const el = document.getElementById(id);
  if (!el) return;
  // Update type
  el.className = 'toast ' + type;
  // Update icon
  const icon = el.querySelector('.toast-icon');
  if (icon) icon.textContent = TOAST_ICONS[type] || 'ℹ';
  // Update title
  const title = el.querySelector('.toast-title');
  if (title) title.textContent = TOAST_TITLES[type] || type;
  // Update message
  const msg = el.querySelector('.toast-msg');
  if (msg) msg.textContent = message;
  // Start auto-dismiss
  el.style.setProperty('--toast-dur', '4000ms');
  const prog = el.querySelector('.toast-progress');
  if (prog) {
    prog.style.animation = 'none';
    void prog.offsetWidth;
    prog.style.animation = 'toastProgress 4s linear forwards';
  }
  setTimeout(function() { _dismissToast(el); }, 4000);
};

/* ══ 14. TABLE MUTATION OBSERVER — auto-stagger new rows ══ */
function initTableObserver() {
  const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(m) {
      if (m.type === 'childList' && m.addedNodes.length) {
        const container = m.target.closest('#recentTable, #gradeTable, #metricsTableSummary');
        if (!container) return;
        const rows = container.querySelectorAll('tbody tr');
        rows.forEach(function(r, i) {
          r.style.animationDelay = (i * 40) + 'ms';
          r.style.animationPlayState = 'running';
        });
      }
    });
  });

  ['recentTable', 'gradeTable'].forEach(function(id) {
    const el = document.getElementById(id);
    if (el) observer.observe(el, { childList: true, subtree: true });
  });

  const mst = document.getElementById('metricsTableSummary');
  if (mst) observer.observe(mst, { childList: true, subtree: true });
}

/* ══ BOOT ══ */
document.addEventListener('DOMContentLoaded', () => {
  initLoader();
  initCursor();
  initParticles();
  initScrollHeader();
  injectRevealClasses();   // [A] must run before initScrollReveal
  initScrollReveal();
  setTimeout(runCounters, 600);
  initRipples();
  patchDrawer();
  patchSwitchTab();
  initTableObserver();
});

})();