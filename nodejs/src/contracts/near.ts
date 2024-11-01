// src/contracts/near.ts
import { connect, keyStores, Contract, WalletConnection } from 'near-api-js';
import { nearConfig } from '../config';
import { TraderContribution } from '../types';
import { logger } from '../utils/logger';

export class NearContracts {
  private capitalPoolContract: Contract;
  private incentivePayoutContract: Contract;
  private wallet: WalletConnection;

  constructor() {
    this.init();
  }

  private async init() {
    try {
      const keyStore = new keyStores.InMemoryKeyStore();
      const near = await connect({
        networkId: nearConfig.networkId,
        keyStore,
        nodeUrl: nearConfig.nodeUrl,
        walletUrl: nearConfig.walletUrl,
        helperUrl: nearConfig.helperUrl,
        headers: {}
      });

      this.wallet = new WalletConnection(near, 'nexus');

      this.capitalPoolContract = new Contract(
        this.wallet.account(),
        nearConfig.capitalPoolContract,
        {
          viewMethods: ['getBalance', 'getTotalCapital', 'getOwner', 'getParticipants'],
          changeMethods: ['contributeCapital', 'withdrawCapital', 'setPortfolioValues']
        }
      );

      this.incentivePayoutContract = new Contract(
        this.wallet.account(),
        nearConfig.incentivePayoutContract,
        {
          viewMethods: ['getContribution', 'getOwedAmount', 'getTotalProfits'],
          changeMethods: ['updateContributions', 'updateProfits', 'withdraw']
        }
      );
    } catch (error) {
      logger.error('Error initializing NEAR contracts:', error);
      throw error;
    }
  }
  // ... rest of the class implementation stays the same
}
  async contributeCapital(amount: string) {
    return await this.capitalPoolContract.contributeCapital({ args: {}, amount });
  }

  async withdrawCapital(amount: string) {
    return await this.capitalPoolContract.withdrawCapital({ amount });
  }

  async updatePortfolioValues(initialValue: string, finalValue: string) {
    return await this.capitalPoolContract.setPortfolioValues({
      initialValue,
      finalValue
    });
  }

  async updateTraderContributions(contributions: TraderContribution[]) {
    return await this.incentivePayoutContract.updateContributions({
      contributions: contributions.map(c => ({
        trader: c.traderId,
        contribution: c.contribution.toString()
      }))
    });
  }

  async updateProfits(newProfits: string) {
    return await this.incentivePayoutContract.updateProfits({ newProfits });
  }

  async getTraderContribution(traderId: string): Promise<string> {
    return await this.incentivePayoutContract.getContribution({ trader: traderId });
  }

  async getOwedAmount(traderId: string): Promise<string> {
    return await this.incentivePayoutContract.getOwedAmount({ trader: traderId });
  }
}