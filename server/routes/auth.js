const router      = require('express').Router();
const jwt         = require('jsonwebtoken');
const User        = require('../models/User');
const VisitorLog  = require('../models/VisitorLog');
const protect     = require('../middleware/auth');
const geoResolve  = require('../config/geoResolve');

// ── JWT helper ────────────────────────────────────────────────────────────────
const sign = (id) => jwt.sign({ id }, process.env.JWT_SECRET, {
  expiresIn: process.env.JWT_EXPIRES_IN,
});

// ── Real IP extractor (handles proxies / Render / Cloudflare) ─────────────────
function getIP(req) {
  return (
    req.headers['x-forwarded-for']?.split(',')[0]?.trim() ||
    req.headers['x-real-ip'] ||
    req.socket?.remoteAddress ||
    'unknown'
  );
}

// ── POST /auth/register ───────────────────────────────────────────────────────
router.post('/register', async (req, res) => {
  const { username, password, role } = req.body;
  if (!username || !password)
    return res.status(400).json({ error: 'username and password required' });
  if (await User.findOne({ username }))
    return res.status(409).json({ error: 'Username already taken' });
  const user = await User.create({ username, password, role });
  res.status(201).json({ token: sign(user._id), username: user.username, role: user.role });
});

// ── POST /auth/login ──────────────────────────────────────────────────────────
// Logs: who, when, from where (IP + geo), browser
router.post('/login', async (req, res) => {
  const { username, password } = req.body;
  if (!username || !password)
    return res.status(400).json({ error: 'username and password required' });

  const user = await User.findOne({ username });

  // ❌ Failed login attempt — log it and stop
  if (!user || !(await user.matchPassword(password))) {
    const ip  = getIP(req);
    const geo = await geoResolve(ip);
    await VisitorLog.create({
      username:  username,
      userId:    user?._id || null,
      role:      user?.role || 'unknown',
      ip,
      userAgent: req.headers['user-agent'] || 'unknown',
      country:   geo.country,
      city:      geo.city,
      isp:       geo.isp,
      action:    'failed_login',
    });
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  // ✅ Successful login — geo-resolve and create log
  const ip  = getIP(req);
  const geo = await geoResolve(ip);

  const log = await VisitorLog.create({
    username:  user.username,
    userId:    user._id,
    role:      user.role,
    ip,
    userAgent: req.headers['user-agent'] || 'unknown',
    country:   geo.country,
    city:      geo.city,
    isp:       geo.isp,
    action:    'login',
  });

  res.json({
    token:    sign(user._id),
    username: user.username,
    role:     user.role,
    logId:    log._id.toString(),   // ← client stores this to call /logout later
  });
});

// ── POST /auth/logout ─────────────────────────────────────────────────────────
// Updates the session log with logoutAt + sessionMs duration
router.post('/logout', protect, async (req, res) => {
  const { logId } = req.body;
  if (logId) {
    try {
      const log = await VisitorLog.findById(logId);
      if (log && log.action === 'login' && !log.logoutAt) {
        log.logoutAt  = new Date();
        log.sessionMs = log.logoutAt - log.loginAt;
        log.action    = 'logout';
        await log.save();
      }
    } catch (_) {}
  }
  res.json({ message: 'Logged out' });
});

// ── POST /auth/logout-beacon ──────────────────────────────────────────────────
// Called by navigator.sendBeacon on tab close (no auth header possible)
// Only needs logId — no token verification, since it only updates an existing doc
router.post('/logout-beacon', async (req, res) => {
  const { logId } = req.body;
  if (logId && logId.length === 24) {
    try {
      const log = await VisitorLog.findById(logId);
      if (log && log.action === 'login' && !log.logoutAt) {
        log.logoutAt  = new Date();
        log.sessionMs = log.logoutAt - log.loginAt;
        log.action    = 'logout';
        await log.save();
      }
    } catch (_) {}
  }
  res.status(204).end();
});

// ── GET /auth/me ──────────────────────────────────────────────────────────────
router.get('/me', protect, async (req, res) => {
  const user = await User.findById(req.user.id).select('-password');
  if (!user) return res.status(404).json({ error: 'User not found' });
  res.json(user);
});

module.exports = router;