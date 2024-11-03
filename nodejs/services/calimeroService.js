const { JsonRpcClient } = require('@calimero-is-near/calimero-p2p-sdk');
const { CALIMERO_CONFIG } = require('../config/calimero');
const { CONTRACT_CONFIG } = require('../config/near');

class CalimeroService {
  constructor() {
    this.client = new JsonRpcClient(
      CALIMERO_CONFIG.nodeUrl,
      CALIMERO_CONFIG.rpcPath
    );
  }

  async registerModelUpdate(traderId, weights) {
    return await this.client.mutate({
      contextId: CONTRACT_CONFIG.FEDERATED_LEARNING_CONTRACT,
      method: "register_model_update",
      argsJson: {
        trader_id: traderId,
        update_id: Date.now().toString(),
        metrics: {
          loss: weights.metrics.loss || 0,
          accuracy: weights.metrics.accuracy || 0,
          epochs_completed: weights.metrics.epochs_completed || 0
        }
      }
    });
  }

  async startAggregation() {
    return await this.client.mutate({
      contextId: CONTRACT_CONFIG.FEDERATED_LEARNING_CONTRACT,
      method: "start_aggregation",
      argsJson: {}
    });
  }

  async finishAggregation(weights) {
    return await this.client.mutate({
      contextId: CONTRACT_CONFIG.FEDERATED_LEARNING_CONTRACT,
      method: "finish_aggregation",
      argsJson: {
        global_model_id: Date.now().toString(),
        validation_metrics: {
          accuracy: weights.metrics.accuracy || 0,
          precision: weights.metrics.precision || 0,
          recall: weights.metrics.recall || 0,
          f1_score: weights.metrics.f1_score || 0
        }
      }
    });
  }

  async getPendingUpdates() {
    return await this.client.query({
      contextId: CONTRACT_CONFIG.FEDERATED_LEARNING_CONTRACT,
      method: "get_pending_updates",
      argsJson: {}
    });
  }

  async getLatestModelId() {
    return await this.client.query({
      contextId: CONTRACT_CONFIG.FEDERATED_LEARNING_CONTRACT,
      method: "get_latest_model_id",
      argsJson: {}
    });
  }

  async getValidationMetrics() {
    return await this.client.query({
      contextId: CONTRACT_CONFIG.FEDERATED_LEARNING_CONTRACT,
      method: "get_validation_metrics",
      argsJson: {}
    });
  }
}

module.exports = new CalimeroService();