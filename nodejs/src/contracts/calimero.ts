// src/contracts/calimero.ts
import { Client } from '@calimero-is-near/calimero-p2p-sdk';
import { calimeroConfig } from '../config';
import { ModelUpdate, GlobalModel } from '../types';
import { logger } from '../utils/logger';

export class CalimeroService {
  private client: Client;
  private modelUpdateCallbacks: ((update: ModelUpdate) => void)[] = [];
  
  constructor() {
    this.client = new Client({
      nodeUrl: calimeroConfig.nodeUrl,
      contextId: calimeroConfig.contextId
    });
    this.subscribeToEvents();
  }
  private async subscribeToEvents() {
    this.client.subscribe((event) => {
      if (event.type === 'NewModelUpdate') {
        this.handleModelUpdate(event.data);
      }
    });
  }

  private async handleModelUpdate(eventData: any) {
    try {
      const modelUpdate: ModelUpdate = {
        traderId: eventData.trader_id,
        timestamp: Date.now(),
        weights: await this.client.storage.get(eventData.update_id),
        metrics: eventData.training_metrics
      };

      this.modelUpdateCallbacks.forEach(callback => callback(modelUpdate));
    } catch (error) {
      logger.error('Error handling model update:', error);
    }
  }

  onModelUpdate(callback: (update: ModelUpdate) => void) {
    this.modelUpdateCallbacks.push(callback);
  }

  async storeModelUpdate(
    traderId: string,
    weights: Buffer,
    metrics: ModelUpdate['metrics']
  ): Promise<string> {
    try {
      // Store weights in Calimero's storage
      const updateId = await this.client.storage.store(weights);

      // Register update in contract
      await this.client.call('register_update', {
        trader_id: traderId,
        update_id: updateId,
        metrics: metrics
      });

      return updateId;
    } catch (error) {
      logger.error('Error storing model update:', error);
      throw error;
    }
  }

  async storeGlobalModel(
    weights: Buffer,
    metrics: GlobalModel['metrics']
  ): Promise<string> {
    try {
      const modelId = await this.client.storage.store(weights);
      
      await this.client.call('finish_aggregation', {
        global_model_id: modelId,
        validation_metrics: metrics
      });

      return modelId;
    } catch (error) {
      logger.error('Error storing global model:', error);
      throw error;
    }
  }

  async getLatestGlobalModel(): Promise<GlobalModel | null> {
    try {
      const modelId = await this.client.call('get_latest_model_id');
      if (!modelId) return null;

      const weights = await this.client.storage.get(modelId);
      const metrics = await this.client.call('get_validation_metrics');

      return {
        modelId,
        timestamp: Date.now(),
        weights,
        metrics
      };
    } catch (error) {
      logger.error('Error getting latest global model:', error);
      throw error;
    }
  }
}