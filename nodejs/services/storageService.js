const fs = require('fs').promises;
const path = require('path');

class StorageService {
  constructor() {
    this.LOCAL_STORAGE_PATH = './storage';
    this.GLOBAL_WEIGHTS_PATH = path.join(this.LOCAL_STORAGE_PATH, 'global_weights');
    this.LOCAL_WEIGHTS_PATH = path.join(this.LOCAL_STORAGE_PATH, 'local_weights');
  }

  async initialize() {
    await fs.mkdir(this.LOCAL_STORAGE_PATH, { recursive: true });
    await fs.mkdir(this.GLOBAL_WEIGHTS_PATH, { recursive: true });
    await fs.mkdir(this.LOCAL_WEIGHTS_PATH, { recursive: true });
  }

  async saveLocalWeights(traderId, weights) {
    // Using local storage for development. In production, this would use Calimero's storage API
    const filename = path.join(this.LOCAL_WEIGHTS_PATH, `trader_${traderId}_weights.json`);
    await fs.writeFile(filename, JSON.stringify(weights));
  }

  async getLocalWeights() {
    // Using local storage for development. In production, this would use Calimero's storage API
    const files = await fs.readdir(this.LOCAL_WEIGHTS_PATH);
    const weights = {};
    
    for (const file of files) {
      const content = await fs.readFile(path.join(this.LOCAL_WEIGHTS_PATH, file), 'utf8');
      weights[file.split('_')[1]] = JSON.parse(content);
    }
    
    return weights;
  }

  async saveGlobalWeights(weights) {
    // Using local storage for development. In production, this would use Calimero's storage API
    const filename = path.join(this.GLOBAL_WEIGHTS_PATH, 'global_weights.json');
    await fs.writeFile(filename, JSON.stringify(weights));
  }

  async getGlobalWeights() {
    // Using local storage for development. In production, this would use Calimero's storage API
    const filename = path.join(this.GLOBAL_WEIGHTS_PATH, 'global_weights.json');
    const content = await fs.readFile(filename, 'utf8');
    return JSON.parse(content);
  }
}

module.exports = new StorageService();