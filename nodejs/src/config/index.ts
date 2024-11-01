// src/config/index.ts
import dotenv from 'dotenv';
import { cleanEnv, str, num } from 'envalid';

dotenv.config();

export const config = cleanEnv(process.env, {
  // Server Configuration
  PORT: num({ default: 3000 }),
  WS_PORT: num({ default: 3001 }),

  // NEAR Configuration
  NEAR_NETWORK_ID: str({ default: 'testnet' }),
  NEAR_NODE_URL: str({ default: 'https://rpc.testnet.near.org' }),
  NEAR_WALLET_URL: str({ default: 'https://wallet.testnet.near.org' }),
  NEAR_HELPER_URL: str({ default: 'https://helper.testnet.near.org' }),
  NEAR_EXPLORER_URL: str({ default: 'https://explorer.testnet.near.org' }),
  
  // Contract addresses
  CAPITAL_POOL_CONTRACT: str(),
  INCENTIVE_PAYOUT_CONTRACT: str(),

  // Calimero Configuration
  CALIMERO_NODE_URL: str(),
  CALIMERO_CONTEXT_ID: str(),
});

export const nearConfig = {
  networkId: config.NEAR_NETWORK_ID,
  nodeUrl: config.NEAR_NODE_URL,
  walletUrl: config.NEAR_WALLET_URL,
  helperUrl: config.NEAR_HELPER_URL,
  explorerUrl: config.NEAR_EXPLORER_URL,
  capitalPoolContract: config.CAPITAL_POOL_CONTRACT,
  incentivePayoutContract: config.INCENTIVE_PAYOUT_CONTRACT,
};

export const calimeroConfig = {
  nodeUrl: config.CALIMERO_NODE_URL,
  contextId: config.CALIMERO_CONTEXT_ID,
};

export const serverConfig = {
  port: config.PORT,
  wsPort: config.WS_PORT,
};