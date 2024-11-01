// src/api/app.ts
import express from 'express';
import cors from 'cors';
import { createModelRoutes } from './routes/model.routes';
import { createTraderRoutes } from './routes/trader.routes';
import { CalimeroService } from '../contracts/calimero';
import { NearContracts } from '../contracts/near';
import { WebSocketServer } from '../websocket/WsServer';
import { logger } from '../utils/logger';

export function createApp(
  calimero: CalimeroService,
  near: NearContracts,
  wsServer: WebSocketServer
) {
  const app = express();

  // Middleware
  app.use(cors());
  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ extended: true, limit: '50mb' }));

  // Routes
  app.use('/api/model', createModelRoutes(calimero, near, wsServer));
  app.use('/api/trader', createTraderRoutes(near, calimero));

  // Error handling
  app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
    logger.error('Unhandled error:', err);
    res.status(500).json({ error: 'Internal server error' });
  });

  return app;
}