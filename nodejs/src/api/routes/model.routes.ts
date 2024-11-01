import { Router } from 'express';
import { CalimeroService } from '../../contracts/calimero';
import { NearContracts } from '../../contracts/near';
import { WebSocketServer } from '../../websocket/WsServer';
import { logger } from '../../utils/logger';
import { ModelUpdate, GlobalModel } from '../../types';

export function createModelRoutes(
  calimero: CalimeroService,
  near: NearContracts,
  wsServer: WebSocketServer
) {
  const router = Router();

  // Submit local model update
  router.post('/update', async (req, res) => {
    try {
      const { traderId, weights, metrics } = req.body;
      
      // Store update in Calimero
      const updateId = await calimero.storeModelUpdate(
        traderId,
        Buffer.from(weights),
        metrics
      );

      // Notify aggregator through WebSocket
      wsServer.sendModelUpdate({
        traderId,
        timestamp: Date.now(),
        weights: Buffer.from(weights),
        metrics
      });

      res.json({ success: true, updateId });
    } catch (error) {
      logger.error('Error handling model update:', error);
      res.status(500).json({ error: 'Failed to store model update' });
    }
  });

  // Get latest global model
  router.get('/global', async (req, res) => {
    try {
      const model = await calimero.getLatestGlobalModel();
      if (!model) {
        return res.status(404).json({ error: 'No global model available' });
      }
      res.json(model);
    } catch (error) {
      logger.error('Error fetching global model:', error);
      res.status(500).json({ error: 'Failed to fetch global model' });
    }
  });

  return router;
}

// src/api/routes/trader.routes.ts
export function createTraderRoutes(
  near: NearContracts,
  calimero: CalimeroService
) {
  const router = Router();

  // Update trader contributions
  router.post('/contributions', async (req, res) => {
    try {
      const { contributions } = req.body;
      await near.updateTraderContributions(contributions);
      res.json({ success: true });
    } catch (error) {
      logger.error('Error updating contributions:', error);
      res.status(500).json({ error: 'Failed to update contributions' });
    }
  });

  // Update profits
  router.post('/profits', async (req, res) => {
    try {
      const { profits } = req.body;
      await near.updateProfits(profits.toString());
      res.json({ success: true });
    } catch (error) {
      logger.error('Error updating profits:', error);
      res.status(500).json({ error: 'Failed to update profits' });
    }
  });

  return router;
}