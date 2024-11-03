require('dotenv').config();

const CALIMERO_CONFIG = {
  nodeUrl: process.env.CALIMERO_NODE_URL,
  rpcPath: "/jsonrpc"
};

module.exports = {
  CALIMERO_CONFIG
};