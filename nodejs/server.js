require('dotenv').config();
const express = require('express');
const nearService = require('./services/nearService');
const storageService = require('./services/storageService');

const weightsRoutes = require('./routes/weights');
const nearRoutes = require('./routes/near');

const app = express();
app.use(express.json());

// Routes
app.use('/weights', weightsRoutes);
app.use('/near', nearRoutes);

// Initialize services and start server
async function initialize() {
  try {
    await Promise.all([
      nearService.initialize(),
      storageService.initialize()
    ]);
    
    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });
  } catch (error) {
    console.error('Failed to initialize server:', error);
    process.exit(1);
  }
}

initialize();