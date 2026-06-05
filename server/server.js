require('dotenv').config();
require('express-async-errors');
const express   = require('express');
const cors      = require('cors');
const morgan    = require('morgan');
const path      = require('path');
const connectDB = require('./config/db');

const authRoutes  = require('./routes/auth');
const proxyRoutes = require('./routes/proxy');

const app  = express();
const PORT = process.env.PORT || 5000;

connectDB();

app.use(cors());
app.use(morgan('dev'));
app.use(express.json());
app.use(express.static(path.join(__dirname, '../frontend')));

app.use('/auth', authRoutes);
app.use('/api',  proxyRoutes);

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/login.html'));
});

app.use((err, req, res, next) => {
  console.error(err.message);
  res.status(500).json({ error: err.message || 'Internal Server Error' });
});

app.listen(PORT, () => {
  console.log(`\nFerroMind running on port ${PORT}`);
  console.log(`Open http://localhost:${PORT}/login.html`)
});