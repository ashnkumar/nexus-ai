const express = require('express');
const router = express.Router();
const calimeroService = require('../services/calimeroService');
const storageService = require('../services/storageService');

router.post('/upload-local-training-weights', async (req, res) => {
  try {
    const { traderId, weights } = req.body;
    
    // Save weights to storage
    await storageService.saveLocalWeights(traderId, weights);
    
    // Register the model update with Calimero
    await calimeroService.registerModelUpdate(traderId, weights);
    
    res.json({ success: true });
  } catch (error) {
    console.error('Error uploading local weights:', error);
    res.status(500).json({ error: error.message });
  }
});

router.get('/retrieve-local-training-weights', async (req, res) => {
  try {
    // Get pending updates from Calimero
    const pendingUpdates = await calimeroService.getPendingUpdates();
    
    // Get weights from storage
    const weights = await storageService.getLocalWeights();
    
    res.json({
      pendingUpdates,
      weights
    });
  } catch (error) {
    console.error('Error retrieving local weights:', error);
    res.status(500).json({ error: error.message });
  }
});

router.post('/upload-global-weights', async (req, res) => {
  try {
    const { weights } = req.body;
    
    // Save weights to storage
    await storageService.saveGlobalWeights(weights);
    
    // Start and finish aggregation on Calimero
    await calimeroService.startAggregation();
    await calimeroService.finishAggregation(weights);
    
    res.json({ success: true });
  } catch (error) {
    console.error('Error uploading global weights:', error);
    res.status(500).json({ error: error.message });
  }
});

router.get('/retrieve-global-model-weights', async (req, res) => {
  try {
    // Get model info from Calimero
    const [latestModelId, metrics] = await Promise.all([
      calimeroService.getLatestModelId(),
      calimeroService.getValidationMetrics()
    ]);
    
    // Get weights from storage
    const weights = await storageService.getGlobalWeights();
    
    res.json({
      modelId: latestModelId,
      metrics,
      weights
    });
  } catch (error) {
    console.error('Error retrieving global weights:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;