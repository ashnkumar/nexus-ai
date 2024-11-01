// src/types/index.ts

export interface ModelUpdate {
  traderId: string;
  timestamp: number;
  weights: Buffer;
  metrics: {
    loss: number;
    accuracy: number;
    epochs: number;
  };
}

export interface GlobalModel {
  modelId: string;
  timestamp: number;
  weights: Buffer;
  metrics: {
    accuracy: number;
    loss: number;
    profitMetrics: {
      avgProfit: number;
      profitStd: number;
    };
  };
}

export interface TraderContribution {
  traderId: string;
  contribution: number;
}

export interface ProfitUpdate {
  timestamp: number;
  profit: number;
  trades: {
    coin: string;
    action: 'BUY' | 'SELL';
    amount: number;
    price: number;
  }[];
}

export interface TraderMetrics {
  accuracy: number;
  profitContribution: number;
  totalTrades: number;
  successfulTrades: number;
}