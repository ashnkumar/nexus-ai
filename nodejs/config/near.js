require('dotenv').config();

const NETWORK_ID = "testnet";
const MY_ACCOUNT = "ashwin-k.testnet";
const NUM_TRADERS = 6;

const NEAR_CONFIG = {
  networkId: NETWORK_ID,
  nodeUrl: "https://rpc.testnet.near.org",
  walletUrl: "https://testnet.mynearwallet.com/",
  helperUrl: "https://helper.testnet.near.org",
  explorerUrl: "https://testnet.nearblocks.io"
};

// Contract addresses
const CONTRACT_CONFIG = {
  CAPITAL_POOLING_CONTRACT: "capital-pooling.testnet", // Replace with actual address
  INCENTIVE_PAYOUT_CONTRACT: "incentive-payout.testnet", // Replace with actual address
  FEDERATED_LEARNING_CONTRACT: "federated-learning.calimero" // Replace with actual address
};

module.exports = {
  NETWORK_ID,
  MY_ACCOUNT,
  NUM_TRADERS,
  NEAR_CONFIG,
  CONTRACT_CONFIG
};