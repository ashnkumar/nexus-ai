// src/websocket/WsServer.ts
import WebSocket from 'ws';
import { logger } from '../utils/logger';
import { ModelUpdate, GlobalModel } from '../types';

export class WebSocketServer {
  private wss: WebSocket.Server;
  private aggregatorConnection: WebSocket | null = null;

  constructor(port: number) {
    this.wss = new WebSocket.Server({ port });
    this.initialize();
  }

  private initialize() {
    this.wss.on('connection', (ws: WebSocket, req: any) => {
      const clientType = req.url?.includes('aggregator') ? 'aggregator' : 'client';
      
      if (clientType === 'aggregator') {
        this.aggregatorConnection = ws;
        logger.info('Aggregator connected to WebSocket');
      }

      ws.on('close', () => {
        if (clientType === 'aggregator') {
          this.aggregatorConnection = null;
          logger.info('Aggregator disconnected from WebSocket');
        }
      });

      ws.on('error', (error) => {
        logger.error('WebSocket error:', error);
      });
    });
  }

  sendModelUpdate(update: ModelUpdate) {
    if (this.aggregatorConnection) {
      try {
        this.aggregatorConnection.send(JSON.stringify({
          type: 'MODEL_UPDATE',
          data: update
        }));
      } catch (error) {
        logger.error('Error sending model update:', error);
      }
    }
  }

  sendGlobalModelUpdate(model: GlobalModel) {
    this.broadcast({
      type: 'GLOBAL_MODEL_UPDATE',
      data: model
    });
  }

  private broadcast(message: any) {
    this.wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(message));
      }
    });
  }
}