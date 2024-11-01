import {
  NearBindgen,
  near,
  call,
  view,
  initialize,
  assert,
  NearPromise,
  UnorderedMap,
  UnorderedSet,
  AccountId,
} from "near-sdk-js";

@NearBindgen({})
export class CapitalPoolingContract {
  static schema = {
    owner: "string",
    incentiveContract: "string",
    totalCapital: "string",
    balances: { class: UnorderedMap, key: "string", value: "string" },
    participants: { class: UnorderedSet, value: "string" },
    initialPortfolioValue: "string",
    finalPortfolioValue: "string",
    profits: "string",
    lastProfitUpdate: "string"
  };

  owner: AccountId;
  incentiveContract: AccountId;
  totalCapital: bigint;
  balances: UnorderedMap<string>;
  participants: UnorderedSet<string>;
  initialPortfolioValue: bigint;
  finalPortfolioValue: bigint;
  profits: bigint;
  lastProfitUpdate: bigint;

  constructor() {
    this.owner = "";
    this.incentiveContract = "";
    this.totalCapital = BigInt(0);
    this.balances = new UnorderedMap<string>("balances-uid");
    this.participants = new UnorderedSet<string>("participants-uid");
    this.initialPortfolioValue = BigInt(0);
    this.finalPortfolioValue = BigInt(0);
    this.profits = BigInt(0);
    this.lastProfitUpdate = BigInt(0);
  }

  @initialize({ privateFunction: true })
  init({ incentiveContract }: { incentiveContract: string }): void {
    this.owner = near.predecessorAccountId();
    this.incentiveContract = incentiveContract;
    near.log(`Contract initialized by owner: ${this.owner} with incentive contract: ${this.incentiveContract}`);
  }

  // Rest of the functions remain the same until setPortfolioValues

  @call({})
  setPortfolioValues({ initialValue, finalValue }: { initialValue: string; finalValue: string }): NearPromise {
    const caller = near.predecessorAccountId();
    near.log(`setPortfolioValues called by ${caller}`);
    assert(caller === this.owner, "Unauthorized");

    this.initialPortfolioValue = BigInt(initialValue);
    this.finalPortfolioValue = BigInt(finalValue);

    // Calculate profits
    let newProfits = BigInt(0);
    if (this.finalPortfolioValue > this.initialPortfolioValue) {
      newProfits = this.finalPortfolioValue - this.initialPortfolioValue;
      this.profits += newProfits;
    }

    near.log(
      `Portfolio values set. Initial: ${this.initialPortfolioValue.toString()}, Final: ${this.finalPortfolioValue.toString()}, New Profits: ${newProfits.toString()}`
    );

    // Update incentive contract with new profits
    if (newProfits > BigInt(0)) {
      return NearPromise.new(this.incentiveContract)
        .functionCall(
          "updateProfits",
          { newProfits: newProfits.toString() },
          BigInt(0),
          near.prepaidGas()
        );
    }

    return NearPromise.new(this.owner);
  }

  // Rest of the functions remain the same
}


  @view({})
  getProfits(): string {
    near.log(`getProfits called`);
    return this.profits.toString();
  }

  @view({})
  getBalance({ traderAddress }: { traderAddress: string }): string {
    near.log(`getBalance called for ${traderAddress}`);
    let balanceString = this.balances.get(traderAddress);
    let balance = balanceString ? BigInt(balanceString) : BigInt(0);
    return balance.toString();
  }

  @view({})
  getTotalCapital(): string {
    near.log(`getTotalCapital called`);
    return this.totalCapital.toString();
  }

  @view({})
  getOwner(): string {
    near.log(`getOwner called`);
    return this.owner;
  }

  @view({})
  getParticipants(): Array<string> {
    near.log(`getParticipants called`);
    return this.participants.toArray();
  }

  @view({})
  getInitialPortfolioValue(): string {
    near.log(`getInitialPortfolioValue called`);
    return this.initialPortfolioValue.toString();
  }

  @view({})
  getFinalPortfolioValue(): string {
    near.log(`getFinalPortfolioValue called`);
    return this.finalPortfolioValue.toString();
  }

  @view({})
  getAllBalances(): Array<{ traderAddress: string; balance: string }> {
    near.log(`getAllBalances called`);
    let balancesArray: Array<{ traderAddress: string; balance: string }> = [];
    return balancesArray;
  }
}