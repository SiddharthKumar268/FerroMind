const mongoose = require('mongoose');

const pageVisitSchema = new mongoose.Schema({
  method:    { type: String },
  path:      { type: String },
  timestamp: { type: Date, default: Date.now },
}, { _id: false });

const visitorLogSchema = new mongoose.Schema({
  // ── Who ──────────────────────────────────────
  username:  { type: String, required: true },
  userId:    { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  role:      { type: String },

  // ── Network ──────────────────────────────────
  ip:        { type: String, default: 'unknown' },
  userAgent: { type: String, default: 'unknown' },

  // ── Geo (resolved server-side) ───────────────
  country:   { type: String, default: null },
  city:      { type: String, default: null },
  isp:       { type: String, default: null },

  // ── Session timing ───────────────────────────
  loginAt:   { type: Date, default: Date.now },
  logoutAt:  { type: Date, default: null },
  sessionMs: { type: Number, default: null },

  // ── Activity ─────────────────────────────────
  pagesVisited: { type: [pageVisitSchema], default: [] },
  action:       { type: String, enum: ['login', 'logout', 'failed_login'], default: 'login' },
});

module.exports = mongoose.model('VisitorLog', visitorLogSchema);
