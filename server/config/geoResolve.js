/**
 * geoResolve.js
 * Resolves an IP address to country, city, and ISP using ip-api.com
 * Free tier — no API key needed. Works for IPv4.
 * Falls back gracefully on localhost / private IPs.
 */

const https = require('https');
const http  = require('http');

const PRIVATE_IP_RE = /^(127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|::1|localhost)/i;

function geoResolve(ip) {
  return new Promise((resolve) => {
    // Skip geo for private/loopback IPs
    if (!ip || PRIVATE_IP_RE.test(ip)) {
      return resolve({ country: 'Local/Private', city: null, isp: null });
    }

    const url = `http://ip-api.com/json/${ip}?fields=status,country,city,isp`;

    http.get(url, { timeout: 4000 }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (parsed.status === 'success') {
            resolve({
              country: parsed.country || null,
              city:    parsed.city    || null,
              isp:     parsed.isp     || null,
            });
          } else {
            resolve({ country: null, city: null, isp: null });
          }
        } catch {
          resolve({ country: null, city: null, isp: null });
        }
      });
    }).on('error', () => resolve({ country: null, city: null, isp: null }))
      .on('timeout', () => resolve({ country: null, city: null, isp: null }));
  });
}

module.exports = geoResolve;
