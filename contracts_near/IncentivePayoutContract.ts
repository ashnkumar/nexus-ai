import {
  NearBindgen,
  near,
  call,
  view,
  initialize,
  assert,
  NearPromise,
  UnorderedMap,
  AccountId,
} from "near-sdk-js";

@NearBindgen({})
export class IncentivePayoutContract {
  // Define schema for serialization
  static schema = {
    owner: "string",
    capitalPoolContract: "string",
    accuracyContributions: { class: UnorderedMap, key: "string", value: "string" }, // trader -> contribution percentage
    owedAmounts: { class: UnorderedMap, key: "string", value: "string" }, // trader -> owed amount
    totalProfits: "string",
    lastDistributionTime: "string",
    isLocked: "boolean"
  };

  owner: AccountId;
  capitalPoolContract: AccountId;
  accuracyContributions: UnorderedMap<string>;  // Stores contribution percentages as strings (0-100)
  owedAmounts: UnorderedMap<string>;  // Stores owed amounts as stringified bigints
  totalProfits: bigint;
  lastDistributionTime: bigint;
  isLocked: boolean;

  constructor() {
    this.owner = "";
    this.capitalPoolContract = "";
    this.accuracyContributions = new UnorderedMap<string>("accuracy-uid");
    this.owedAmounts = new UnorderedMap<string>("owed-uid");
    this.totalProfits = BigInt(0);
    this.lastDistributionTime = BigInt(0);
    this.isLocked = false;
  }

  @initialize({ privateFunction: true })
  init({ capitalPoolContract }: { capitalPoolContract: string }): void {
    this.owner = near.predecessorAccountId();
    this.capitalPoolContract = capitalPoolContract;
    near.log(`Contract initialized by owner: ${this.owner} with capital pool: ${this.capitalPoolContract}`);
  }

  @call({})
  updateContributions({ contributions }: { 
    contributions: Array<{ trader: string; contribution: string }> 
  }): void {
    // Only owner (NodeJS server) can update contributions
    assert(near.predecessorAccountId() === this.owner, "Unauthorized");
    assert(!this.isLocked, "Contract is locked during distribution");
    
    // Validate contributions sum to 100
    let total = 0;
    for (const { contribution } of contributions) {
      total += parseFloat(contribution);
    }
    assert(Math.abs(total - 100) < 0.01, "Contributions must sum to 100");

    // Update contributions
    for (const { trader, contribution } of contributions) {
      this.accuracyContributions.set(trader, contribution);
      near.log(`Updated contribution for ${trader}: ${contribution}%`);
    }
  }

  @call({})
  updateProfits({ newProfits }: { newProfits: string }): void {
    // Only capital pool contract or owner can update profits
    const caller = near.predecessorAccountId();
    assert(
      caller === this.capitalPoolContract || caller === this.owner,
      "Unauthorized"
    );
    
    const profitsToAdd = BigInt(newProfits);
    assert(profitsToAdd >= BigInt(0), "Profits cannot be negative");

    // Lock contract during distribution
    this.isLocked = true;

    try {
      this.totalProfits += profitsToAdd;
      
      // Distribute new profits according to contributions
      for (const trader of this.accuracyContributions.keys()) {
        const contribution = parseFloat(this.accuracyContributions.get(trader)!);
        const share = (profitsToAdd * BigInt(Math.floor(contribution * 100))) / BigInt(10000);
        
        // Update owed amounts
        const currentOwed = this.owedAmounts.get(trader);
        const newOwed = (currentOwed ? BigInt(currentOwed) : BigInt(0)) + share;
        this.owedAmounts.set(trader, newOwed.toString());
        
        near.log(`Updated owed amount for ${trader}: ${newOwed.toString()}`);
      }

      this.lastDistributionTime = near.blockTimestamp();
    } finally {
      this.isLocked = false;
    }
  }

  @call({})
  withdraw(): NearPromise {
    const trader = near.predecessorAccountId();
    const owedString = this.owedAmounts.get(trader);
    assert(owedString, "No funds to withdraw");
    
    const owed = BigInt(owedString);
    assert(owed > BigInt(0), "No funds to withdraw");

    // Reset owed amount
    this.owedAmounts.set(trader, "0");
    
    near.log(`${trader} withdrew ${owed.toString()}`);
    return NearPromise.new(trader).transfer(owed);
  }

  @view({})
  getContribution({ trader }: { trader: string }): string {
    const contribution = this.accuracyContributions.get(trader);
    return contribution || "0";
  }

  @view({})
  getOwedAmount({ trader }: { trader: string }): string {
    const owed = this.owedAmounts.get(trader);
    return owed || "0";
  }

  @view({})
  getTotalProfits(): string {
    return this.totalProfits.toString();
  }

  @view({})
  getAllContributions(): Array<{ trader: string; contribution: string }> {
    let contributions: Array<{ trader: string; contribution: string }> = [];
    for (const trader of this.accuracyContributions.keys()) {
      contributions.push({
        trader,
        contribution: this.accuracyContributions.get(trader)!
      });
    }
    return contributions;
  }

  @view({})
  getLastDistributionTime(): string {
    return this.lastDistributionTime.toString();
  }
}