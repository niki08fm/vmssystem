

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const User = require('./models/User');
const Visitor = require('./models/Visitor');

const app = express();
const JWT_SECRET = '334231'; 

app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '10mb' }));

// --- DB CONNECTION ---
mongoose.connect('mongodb+srv://b02206370:Manu08fm.@cluster0.jhcep.mongodb.net/gms')

  .then(async () => {
    console.log("MongoDB Connected");
    const adminExists = await User.findOne({ username: 'admin' });
    if (!adminExists) {
      await User.create({ username: 'admin', password: 'admin123', role: 'ADMIN' });
      console.log("Default Admin Created");
    }
  })
  .catch(err => console.error(err));

// --- MIDDLEWARE ---
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  if (!token) return res.json({ success: false, message: "Access Denied" });
  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) return res.json({ success: false, message: "Invalid Token" });
    req.user = user;
    next();
  });
};

// --- ROUTES ---

// LOGIN
app.post('/api/login', async (req, res) => {
  const { username, password } = req.body;
  try {
    const user = await User.findOne({ username });
    if (!user || user.password !== password) {
      return res.json({ success: false, message: "Invalid Credentials" });
    }
    const token = jwt.sign({ id: user._id, role: user.role }, JWT_SECRET, { expiresIn: '12h' });
    res.json({ success: true, token, role: user.role, username: user.username });
  } catch (e) { res.json({ success: false, message: "Server Error" }); }
});

// ADMIN: GET STATS
app.get('/api/admin/stats', authenticateToken, async (req, res) => {
  if(req.user.role !== 'ADMIN') return res.json({ success: false });
  try {
    const total = await Visitor.countDocuments();
    const inside = await Visitor.countDocuments({ status: 'INSIDE' });
    const exited = await Visitor.countDocuments({ status: 'EXITED' });
    const pending = await Visitor.countDocuments({ status: 'PENDING' });
    res.json({ success: true, stats: { total, inside, exited, pending } });
  } catch (e) { res.json({ success: false }); }
});

// ADMIN: GET ALL WATCHMEN
app.get('/api/admin/watchmen', authenticateToken, async (req, res) => {
  if(req.user.role !== 'ADMIN') return res.json({ success: false });
  try {
    const watchmen = await User.find({ role: 'WATCHMAN' }).select('-password').sort({ createdAt: -1 });
    res.json({ success: true, watchmen });
  } catch (e) { res.json({ success: false }); }
});

// ADMIN: ADD WATCHMAN
app.post('/api/admin/add-watchman', authenticateToken, async (req, res) => {
  if(req.user.role !== 'ADMIN') return res.json({ success: false });
  try {
    await User.create({ ...req.body, role: 'WATCHMAN' });
    res.json({ success: true });
  } catch (e) { res.json({ success: false, message: "Username taken" }); }
});

// ADMIN: DELETE WATCHMAN
app.delete('/api/admin/watchman/:id', authenticateToken, async (req, res) => {
  if(req.user.role !== 'ADMIN') return res.json({ success: false });
  try {
    await User.findByIdAndDelete(req.params.id);
    res.json({ success: true });
  } catch (e) { res.json({ success: false }); }
});

// ADMIN: HISTORY LOGS (Pagination + Filters)
app.get('/api/admin/history', authenticateToken, async (req, res) => {
  if(req.user.role !== 'ADMIN') return res.json({ success: false });

  const { page = 1, limit = 10, search, startDate, endDate } = req.query;
  const query = {};

  // Search Filter (Regex for Name, Host, or Plate)
  if (search) {
    query.$or = [
      { guestName: { $regex: search, $options: 'i' } },
      { hostName: { $regex: search, $options: 'i' } },
      { plateNumber: { $regex: search, $options: 'i' } }
    ];
  }

  // Date Range Filter
  if (startDate || endDate) {
    query.createdAt = {};
    if (startDate) query.createdAt.$gte = new Date(startDate);
    if (endDate) {
      const end = new Date(endDate);
      end.setHours(23, 59, 59, 999); // Include the whole end day
      query.createdAt.$lte = end;
    }
  }

  try {
    const logs = await Visitor.find(query)
      .sort({ createdAt: -1 }) // Newest first
      .limit(limit * 1)
      .skip((page - 1) * limit);

    const count = await Visitor.countDocuments(query);

    res.json({
      success: true,
      logs,
      totalPages: Math.ceil(count / limit),
      currentPage: parseInt(page),
      totalRecords: count
    });
  } catch (e) { res.json({ success: false, message: "Error fetching history" }); }
});

// PUBLIC & WATCHMAN ROUTES
app.post('/api/guest/register', async (req, res) => {
  const entryCode = Math.floor(100000 + Math.random() * 900000).toString();
  await Visitor.create({ ...req.body, entryCode });
  res.json({ success: true, entryCode });
});

app.post('/api/watchman/verify', authenticateToken, async (req, res) => {
  const visitor = await Visitor.findOne({ entryCode: req.body.code, status: 'PENDING' });
  if (visitor) res.json({ success: true, visitor });
  else res.json({ success: false, message: "Invalid Code" });
});

app.post('/api/watchman/entry', authenticateToken, async (req, res) => {
  try {
    const { visitorId, plateNumber, plateImage, originalImage } = req.body;
    await Visitor.findByIdAndUpdate(visitorId, {
      plateNumber,
      plateImage: plateImage || null,
      originalImage: originalImage || null,
      status: 'INSIDE',
      entryTime: new Date()
    });
    res.json({ success: true });
  } catch (e) {
    console.error(e);
    res.json({ success: false, message: "Server error during entry" });
  }
});

app.post('/api/watchman/exit', authenticateToken, async (req, res) => {
  const visitor = await Visitor.findOne({ plateNumber: req.body.plateNumber, status: 'INSIDE' });
  if (visitor) {
    visitor.status = 'EXITED';
    visitor.exitTime = new Date();
    await visitor.save();
    res.json({ success: true, visitor });
  } else {
    res.json({ success: false, message: "No entry found" });
  }
});

app.get('/api/guest/status/:code', async (req, res) => {
  const visitor = await Visitor.findOne({ entryCode: req.params.code });
  if (visitor) res.json({ success: true, status: visitor.status });
  else res.json({ success: false });
});

app.get('/',(req,res) =>{
  return "hello";
})

app.listen(3000, '0.0.0.0', () => console.log("Backend running on 3000"));