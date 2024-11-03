const express = require('express');
const router = express.Router();
const nearService = require('../services/nearService');
const { NUM_TRADERS } = require('../config/near');

router.post('/seed-near-info', async (req, res) => {
  try {
    const traderAccounts = [];
    
    // Create and fund accounts for each trader
    for (let i = 0; i < NUM_TRADERS; i++) {
      const traderAccountId = nearService.getTraderAccountId(i);
      
      // Create and fund account
      const accountInfo = await nearService.createTraderAccount(traderAccountId);
      
      // Contribute capital
      await nearService.contributeCapital(traderAccountId);
      
      traderAccounts.push(accountInfo);
    }
    
    // Update contributions (equal distribution for initial setup)
    const contributions = traderAccounts.map(({ traderAccountId }) => ({
      trader: traderAccountId,
      contribution: (100 / NUM_TRADERS).toString()
    }));
    
    await nearService.updateContributions(contributions);
    
    res.json({ success: true, traderAccounts });
  } catch (error) {
    console.error('Error seeding NEAR info:', error);
    res.status(500).json({ error: error.message });
  }
});

router.post('/update-near-profits', async (req, res) => {
  try {
    const { profits } = req.body;
    await nearService.updateProfits(profits);
    res.json({ success: true });
  } catch (error) {
    console.error('Error updating profits:', error);
    res.status(500).json({ error: error.message });
  }
});

router.post('/distribute-near-profits', async (req, res) => {
  try {
    const distributionResults = [];
    
    // Distribute profits to each trader
    for (let i = 0; i < NUM_TRADERS; i++) {
      const traderAccountId = nearService.getTraderAccountId(i);
      
      try {
        await nearService.withdrawTraderProfits(traderAccountId);
        distributionResults.push({
          trader: traderAccountId,
          success: true
        });
      } catch (error) {
        distributionResults.push({
          trader: traderAccountId,
          success: false,
          error: error.message
        });
      }
    }
    
    res.json({ success: true, distributions: distributionResults });
  } catch (error) {
    console.error('Error distributing profits:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;