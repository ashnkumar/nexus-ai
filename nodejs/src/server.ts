// src/server.ts
import { serverConfig } from './config';
import { createApp } from './api/app';
import { CalimeroService } from './contracts/calimero';
import { NearContracts } from './contracts/near';
import { WebSocketServer } from './websocket/WsServer';
import { logger } from './utils/logger';

async function startServer() {
  try {
    const calimero = new CalimeroService();
    const near = new NearContracts();
    const wsServer = new WebSocketServer(serverConfig.wsPort);

    calimero.onModelUpdate((update) => {
      wsServer.sendModelUpdate(update);
    });

    const app = createApp(calimero, near, wsServer);
    
    app.listen(serverConfig.port, () => {
      logger.info(`Server running on port ${serverConfig.port}`);
      logger.info(`WebSocket server running on port ${serverConfig.wsPort}`);
    });

  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();