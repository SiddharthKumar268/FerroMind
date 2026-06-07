/**
 * pageTracker.js
 * Middleware that appends the current request (method + path) to the
 * active VisitorLog session stored in MongoDB Atlas.
 *
 * Attach AFTER the `protect` middleware on any route you want tracked.
 * The logId is passed by the client in the Authorization header's decoded JWT
 * via req.user.logId (set during login).
 *
 * Usage in server.js:
 *   const pageTracker = require('./middleware/pageTracker');
 *   app.use('/api', protect, pageTracker, proxyRoutes);
 */

const VisitorLog = require('../models/VisitorLog');

const pageTracker = async (req, res, next) => {
  try {
    const logId = req.headers['x-log-id'];  // client sends this header
    if (logId && logId.length === 24) {
      // Fire-and-forget — don't block the request
      VisitorLog.findByIdAndUpdate(logId, {
        $push: {
          pagesVisited: {
            method:    req.method,
            path:      req.path,
            timestamp: new Date(),
          },
        },
      }).exec().catch(() => {});
    }
  } catch (_) {}
  next();
};

module.exports = pageTracker;
