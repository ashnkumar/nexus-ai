const { connect, keyStores, KeyPair, utils } = require('near-api-js');
const { NEAR_CONFIG, NETWORK_ID, MY_ACCOUNT, CONTRACT_CONFIG } = require('../config/near');

class NearService {
  constructor() {
    this.connection = null;
    this.accountObj = null;
  }

  async initialize() {
    const keyStore = new keyStores.InMemoryKeyStore();
    const privateKey = process.env.NEAR_PRIVATE_KEY;
    const keyPair = KeyPair.fromString(privateKey);
    await keyStore.setKey(NETWORK_ID, MY_ACCOUNT, keyPair);

    this.connection = await connect({
      ...NEAR_CONFIG,
      keyStore,
    });

    this.accountObj = await this.connection.account(MY_ACCOUNT);
  }

  getTraderAccountId(index) {
    return `trader${index}.${MY_ACCOUNT}`;
  }

  async createTraderAccount(traderAccountId) {
    const keyPair = KeyPair.fromRandom('ed25519');
    const publicKey = keyPair.publicKey.toString();
    
    await this.accountObj.createAccount(
      traderAccountId,
      publicKey,
      utils.format.parseNearAmount('5') // Fund with 5 NEAR
    );

    return { traderAccountId, keyPair };
  }

  async contributeCapital(traderAccountId) {
    const traderAccount = await this.connection.account(traderAccountId);
    await traderAccount.functionCall({
      contractId: CONTRACT_CONFIG.CAPITAL_POOLING_CONTRACT,
      methodName: 'contributeCapital',
      args: {},
      attachedDeposit: utils.format.parseNearAmount('2') // Contribute 2 NEAR
    });
  }

  async updateContributions(contributions) {
    await this.accountObj.functionCall({
      contractId: CONTRACT_CONFIG.INCENTIVE_PAYOUT_CONTRACT,
      methodName: 'updateContributions',
      args: { contributions }
    });
  }

  async updateProfits(profits) {
    await this.accountObj.functionCall({
      contractId: CONTRACT_CONFIG.INCENTIVE_PAYOUT_CONTRACT,
      methodName: 'updateProfits',
      args: { newProfits: profits.toString() }
    });
  }

  async withdrawTraderProfits(traderAccountId) {
    const traderAccount = await this.connection.account(traderAccountId);
    await traderAccount.functionCall({
      contractId: CONTRACT_CONFIG.INCENTIVE_PAYOUT_CONTRACT,
      methodName: 'withdraw',
      args: {}
    });
  }
}

module.exports = new NearService();