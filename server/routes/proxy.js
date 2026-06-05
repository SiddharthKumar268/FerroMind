/**
 * proxy.js — FerroMind API Routes (v2 — Performance Optimised)
 *
 * Key changes from v1:
 *   1. Persistent Python REPL process — loaded once, stays alive
 *   2. Batch predictions (analytics / recent send all heats in one call)
 *   3. CSV data cached in memory with mtime checks
 *   4. Analytics response cached with TTL
 *   5. Timestamps added to all responses (data_loaded_at, served_at)
 */

const router     = require('express').Router();
const protect    = require('../middleware/auth');
const { spawn }  = require('child_process');
const path       = require('path');
const fs         = require('fs');
const readline   = require('readline');

const PREDICT_PY   = path.join(__dirname, '../../src/predict.py');
const TEST_CSV     = path.join(__dirname, '../../data/processed/sms_test.csv');
const METRICS_CSV  = path.join(__dirname, '../../reports/metrics_summary.csv');
const GRADE_CSV    = path.join(__dirname, '../../reports/per_grade_rmse.csv');
const TARGET_COLS  = ['Mn', 'Si', 'C', 'S', 'P'];

// ─────────────────────────────────────────────
// CSV CACHE — parse once, re-parse only on change
// ─────────────────────────────────────────────
const _csvCache = {};

function parseCSVCached(filepath) {
  const stat = fs.existsSync(filepath) ? fs.statSync(filepath) : null;
  if (!stat) return Promise.reject(new Error(`File not found: ${filepath}`));

  const mtime = stat.mtimeMs;
  if (_csvCache[filepath] && _csvCache[filepath].mtime === mtime) {
    return Promise.resolve(_csvCache[filepath].rows);
  }

  return new Promise((resolve, reject) => {
    const rows = [];
    const rl   = readline.createInterface({
      input: fs.createReadStream(filepath),
      crlfDelay: Infinity,
    });

    let headers = null;
    rl.on('line', (line) => {
      const cols = line.split(',');
      if (!headers) { headers = cols; return; }
      const row = {};
      headers.forEach((h, i) => { row[h.trim()] = cols[i]?.trim(); });
      rows.push(row);
    });

    rl.on('close', () => {
      _csvCache[filepath] = { rows, mtime, loadedAt: new Date().toISOString() };
      resolve(rows);
    });
    rl.on('error', reject);
  });
}

function getCSVLoadedAt(filepath) {
  return _csvCache[filepath]?.loadedAt || null;
}


// ─────────────────────────────────────────────
// PERSISTENT PYTHON REPL PROCESS
// ─────────────────────────────────────────────
let _py = null;
let _pyReady = false;
let _pyStartedAt = null;
let _pendingCallbacks = [];  // queue of {resolve, reject}
let _pyBuffer = '';

function startPythonREPL() {
  console.log('[proxy] Starting persistent Python REPL process...');
  _py = spawn('python', [PREDICT_PY, '--mode', 'repl'], {
    cwd: path.join(__dirname, '../..'),
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  _py.stderr.on('data', (data) => {
    console.log('[predict.py]', data.toString().trim());
  });

  _py.stdout.on('data', (chunk) => {
    _pyBuffer += chunk.toString();
    // Process complete lines
    let lines = _pyBuffer.split('\n');
    _pyBuffer = lines.pop();  // keep incomplete line in buffer

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const parsed = JSON.parse(line.trim());

        // First message is the "ready" signal
        if (!_pyReady && parsed.ready) {
          _pyReady = true;
          _pyStartedAt = new Date().toISOString();
          console.log('[proxy] Python REPL ready!');
          continue;
        }

        // Route response to first pending callback
        if (_pendingCallbacks.length > 0) {
          const { resolve } = _pendingCallbacks.shift();
          resolve(parsed);
        }
      } catch (e) {
        console.error('[proxy] Failed to parse Python response:', line, e.message);
        if (_pendingCallbacks.length > 0) {
          const { reject } = _pendingCallbacks.shift();
          reject(new Error(`Parse error: ${e.message}`));
        }
      }
    }
  });

  _py.on('close', (code) => {
    console.error(`[proxy] Python REPL exited with code ${code}. Restarting in 2s...`);
    _pyReady = false;
    // Reject all pending callbacks
    while (_pendingCallbacks.length > 0) {
      const { reject } = _pendingCallbacks.shift();
      reject(new Error('Python process crashed'));
    }
    setTimeout(startPythonREPL, 2000);
  });

  _py.on('error', (err) => {
    console.error('[proxy] Python REPL error:', err.message);
  });
}

// Start immediately on module load
startPythonREPL();


function callPython(payload, timeoutMs = 120000) {
  return new Promise((resolve, reject) => {
    if (!_pyReady || !_py) {
      return reject(new Error('Python model not ready yet — please wait'));
    }

    const timer = setTimeout(() => {
      // Remove from queue
      const idx = _pendingCallbacks.findIndex(cb => cb._timer === timer);
      if (idx >= 0) _pendingCallbacks.splice(idx, 1);
      reject(new Error('Python call timed out'));
    }, timeoutMs);

    const cb = { resolve: (val) => { clearTimeout(timer); resolve(val); },
                 reject:  (err) => { clearTimeout(timer); reject(err);  },
                 _timer: timer };
    _pendingCallbacks.push(cb);

    _py.stdin.write(JSON.stringify(payload) + '\n');
  });
}


// ─────────────────────────────────────────────
// ANALYTICS CACHE
// ─────────────────────────────────────────────
let _analyticsCache = null;
let _analyticsCacheTime = 0;
const ANALYTICS_TTL = 60000; // 60 seconds


// ─────────────────────────────────────────────
// RATE LIMITER
// ─────────────────────────────────────────────
const _calls = {};
const rateLimit = (max, windowMs) => (req, res, next) => {
  const ip  = req.ip;
  const now = Date.now();
  _calls[ip] = (_calls[ip] || []).filter(t => now - t < windowMs);
  if (_calls[ip].length >= max)
    return res.status(429).json({ error: 'Too many requests' });
  _calls[ip].push(now);
  next();
};


// ─────────────────────────────────────────────
// ROUTES
// ─────────────────────────────────────────────

router.get('/status', protect, async (req, res) => {
  const modelsDir = path.join(__dirname, '../../models');
  const modelFiles = fs.existsSync(modelsDir)
    ? fs.readdirSync(modelsDir).filter(f => f.endsWith('.pkl')).map(f => f.replace('.pkl', ''))
    : [];

  let testRows = 0;
  let gradeCount = 0;
  try {
    const rows  = await parseCSVCached(TEST_CSV);
    testRows    = rows.length;
    const grades = new Set(rows.map(r => r['BOF Grade Code']).filter(Boolean));
    gradeCount  = grades.size;
  } catch (_) {}

  res.json({
    status       : 'ok',
    models_loaded: modelFiles,
    test_rows    : testRows,
    grade_count  : gradeCount,
    python_ready : _pyReady,
    python_started_at: _pyStartedAt,
    data_loaded_at: getCSVLoadedAt(TEST_CSV),
    served_at    : new Date().toISOString(),
  });
});


router.get('/grades', protect, async (req, res) => {
  const rows   = await parseCSVCached(TEST_CSV);
  const grades = [...new Set(rows.map(r => r['BOF Grade Code']).filter(Boolean))].sort();
  res.json({
    grades,
    data_loaded_at: getCSVLoadedAt(TEST_CSV),
    served_at: new Date().toISOString(),
  });
});


router.get('/metrics', protect, async (req, res) => {
  if (!fs.existsSync(METRICS_CSV))
    return res.status(404).json({ error: 'metrics_summary.csv not found — run evaluate.py first' });

  const rows   = await parseCSVCached(METRICS_CSV);
  const result = {};

  for (const row of rows) {
    const model = (row['model'] || row['Model'] || 'unknown').trim();
    const elem  = (row['element'] || row['Element'] || '?').trim();
    if (!result[model]) result[model] = {};
    result[model][elem] = {
      rmse: parseFloat(row['rmse'] || row['RMSE'] || 0),
      mae : parseFloat(row['mae']  || row['MAE']  || 0),
      r2  : parseFloat(row['r2']   || row['R2']   || 0),
    };
  }

  let perGrade = [];
  if (fs.existsSync(GRADE_CSV)) {
    const pgRows = await parseCSVCached(GRADE_CSV);
    perGrade = pgRows.map(r => ({
      grade  : r['Grade'],
      n_heats: parseInt(r['N_heats']),
      rmse_mn: parseFloat(r['RMSE_Mn']),
      rmse_si: parseFloat(r['RMSE_Si']),
      rmse_c : parseFloat(r['RMSE_C']),
      rmse_s : parseFloat(r['RMSE_S']),
      rmse_p : parseFloat(r['RMSE_P']),
    }));
  }

  res.json({
    model_metrics: result,
    per_grade: perGrade,
    data_loaded_at: getCSVLoadedAt(METRICS_CSV),
    served_at: new Date().toISOString(),
  });
});


// ── RECENT — batch predict all heats in one call ──
router.get('/recent', protect, async (req, res) => {
  const n    = parseInt(req.query.n) || 20;
  const rows = await parseCSVCached(TEST_CSV);
  const recent = rows.slice(-n);

  // Build all heat objects
  const heats = recent.map(row => {
    const heat = {};
    for (const [k, v] of Object.entries(row)) {
      const num = parseFloat(v);
      if (!isNaN(num)) heat[k] = num;
    }
    return heat;
  });

  // Single batch prediction call instead of N sequential spawns
  let compositions = [];
  try {
    const r = await callPython({ mode: 'batch_predict', heats });
    if (r.ok) compositions = r.compositions;
  } catch (e) {
    console.error('[recent] Batch predict error:', e.message);
  }

  const results = recent.map((row, i) => {
    const predicted = compositions[i] || Object.fromEntries(TARGET_COLS.map(e => [e, null]));
    const actual = Object.fromEntries(
      TARGET_COLS.map(e => [e, row[e] ? parseFloat(row[e]) : null])
    );
    return {
      heat_number: row['Heat_Number'] || '',
      grade      : row['BOF Grade Code'] || '',
      actual,
      predicted,
    };
  });

  res.json({
    heats: results,
    count: results.length,
    data_loaded_at: getCSVLoadedAt(TEST_CSV),
    served_at: new Date().toISOString(),
  });
});


// ── ANALYTICS — batch predict + caching ──
router.get('/analytics', protect, async (req, res) => {
  // Return cached if fresh
  const now = Date.now();
  if (_analyticsCache && (now - _analyticsCacheTime) < ANALYTICS_TTL) {
    return res.json({
      ..._analyticsCache,
      served_at: new Date().toISOString(),
      cached: true,
    });
  }

  const t0 = Date.now();
  const result = {};

  // 1. Grade distribution
  try {
    const rows        = await parseCSVCached(TEST_CSV);
    result.total_heats = rows.length;

    const gradeCounts = {};
    for (const r of rows) {
      const g = r['BOF Grade Code'];
      if (g) gradeCounts[g] = (gradeCounts[g] || 0) + 1;
    }
    const sorted = Object.entries(gradeCounts).sort((a, b) => b[1] - a[1]).slice(0, 10);
    result.grade_dist = {
      labels: sorted.map(([g]) => g),
      counts: sorted.map(([, c]) => c),
    };

    // FA vs Mn scatter (sample 200)
    const scatter = rows
      .filter(r => r['Total_FA_Ladle'] && r['Mn'])
      .map(r => ({ x: parseFloat(r['Total_FA_Ladle']), y: parseFloat(r['Mn']) }))
      .filter(p => !isNaN(p.x) && !isNaN(p.y));
    const sampled = scatter.sort(() => 0.5 - Math.random()).slice(0, 200);
    result.fa_scatter = {
      x: sampled.map(p => p.x),
      y: sampled.map(p => p.y),
    };

    // Element stats
    result.element_stats = {};
    for (const elem of TARGET_COLS) {
      const vals = rows.map(r => parseFloat(r[elem])).filter(v => !isNaN(v));
      if (!vals.length) continue;
      vals.sort((a, b) => a - b);
      const mean   = vals.reduce((s, v) => s + v, 0) / vals.length;
      const std    = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length);
      result.element_stats[elem] = {
        mean  : +mean.toFixed(4),
        std   : +std.toFixed(4),
        min   : +vals[0].toFixed(4),
        max   : +vals[vals.length - 1].toFixed(4),
        median: +vals[Math.floor(vals.length / 2)].toFixed(4),
      };
    }

    // ── Recent accuracy — BATCH predict last 50 heats ──
    const recent50 = rows.slice(-50);
    const heats = recent50.map(row => {
      const heat = {};
      for (const [k, v] of Object.entries(row)) {
        const num = parseFloat(v);
        if (!isNaN(num)) heat[k] = num;
      }
      return heat;
    });

    const labels = [], actual = [], predicted = [];
    try {
      const r = await callPython({ mode: 'batch_predict', heats });
      if (r.ok && r.compositions) {
        r.compositions.forEach((comp, i) => {
          labels.push(recent50[i]['Heat_Number'] || '');
          actual.push(parseFloat(recent50[i]['Mn']) || null);
          predicted.push(comp['Mn']);
        });
      }
    } catch (e) {
      console.error('[analytics] Batch predict error:', e.message);
    }
    result.recent_accuracy = { labels, actual, predicted };

  } catch (e) {
    result.error = e.message;
  }

  // 2. Metrics from CSV
  result.r2_compare   = { elements: TARGET_COLS, models: {} };
  result.xgb_rmse     = { elements: TARGET_COLS, rmse: [] };
  result.metrics_table = [];
  result.best_model   = { name: 'N/A', mn_r2: 0 };

  if (fs.existsSync(METRICS_CSV)) {
    try {
      const rows  = await parseCSVCached(METRICS_CSV);
      const byModel = {};
      for (const row of rows) {
        const m = (row['Model'] || row['model'] || '').trim();
        const e = (row['Element'] || row['element'] || '').trim();
        if (!byModel[m]) byModel[m] = {};
        byModel[m][e] = {
          r2  : parseFloat(row['R2']   || row['r2']   || 0),
          rmse: parseFloat(row['RMSE'] || row['rmse'] || 0),
          mae : parseFloat(row['MAE']  || row['mae']  || 0),
        };
      }

      for (const [m, elems] of Object.entries(byModel)) {
        const display = m.replace('_composition', '').replace(/_/g, ' ')
          .replace(/\b\w/g, c => c.toUpperCase());
        result.r2_compare.models[display] = TARGET_COLS.map(e => +(elems[e]?.r2 || 0).toFixed(4));
        const mnR2 = elems['Mn']?.r2 || 0;
        if (mnR2 > result.best_model.mn_r2)
          result.best_model = { name: display, mn_r2: +mnR2.toFixed(4) };
      }

      const xgb = byModel[Object.keys(byModel).find(k => k.includes('xgboost')) || ''] || {};
      result.xgb_rmse.rmse = TARGET_COLS.map(e => +(xgb[e]?.rmse || 0).toFixed(5));

      for (const elem of TARGET_COLS) {
        const d = xgb[elem];
        if (!d) continue;
        const diff = d.r2 >= 0.9 ? 'Easy' : d.r2 >= 0.5 ? 'Medium' : d.r2 >= 0.05 ? 'Hard' : 'Hardest';
        result.metrics_table.push({
          el  : elem,
          rmse: d.rmse.toFixed(5),
          mae : d.mae.toFixed(5),
          r2  : d.r2.toFixed(4),
          diff,
        });
      }
    } catch (_) {}
  }

  // 3. Per-grade RMSE top 10
  result.per_grade_rmse = { grades: [], n_heats: [], rmse_mn: [] };
  if (fs.existsSync(GRADE_CSV)) {
    try {
      const pgRows = (await parseCSVCached(GRADE_CSV)).slice(0, 10);
      result.per_grade_rmse = {
        grades : pgRows.map(r => r['Grade']),
        n_heats: pgRows.map(r => parseInt(r['N_heats'])),
        rmse_mn: pgRows.map(r => parseFloat(r['RMSE_Mn'])),
      };
    } catch (_) {}
  }

  // Add timing and dates
  const elapsed = Date.now() - t0;
  result.data_loaded_at = getCSVLoadedAt(TEST_CSV);
  result.computed_at = new Date().toISOString();
  result.served_at = new Date().toISOString();
  result.compute_time_ms = elapsed;
  result.cached = false;

  // Cache the result
  _analyticsCache = result;
  _analyticsCacheTime = now;

  res.json(result);
});


// ── PREDICT — single heat via persistent process ──
router.post('/predict', protect, async (req, res) => {
  const { heat } = req.body;
  if (!heat) return res.status(400).json({ error: '`heat` object required' });
  const result = await callPython({ mode: 'predict', heat });
  if (!result.ok) return res.status(500).json({ error: result.error });
  res.json({ ...result, served_at: new Date().toISOString() });
});


// ── OPTIMISE — vectorized via persistent process ──
router.post('/optimise', protect, rateLimit(10, 60_000), async (req, res) => {
  const { heat, targets, step } = req.body;
  if (!heat)    return res.status(400).json({ error: '`heat` object required' });
  if (!targets) return res.status(400).json({ error: '`targets` object required' });
  const result = await callPython({ mode: 'optimise', heat, targets, step: step || 100 });
  if (!result.ok) return res.status(500).json({ error: result.error });
  res.json({ ...result, served_at: new Date().toISOString() });
});


module.exports = router;