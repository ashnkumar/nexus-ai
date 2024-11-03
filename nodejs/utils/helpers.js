const { utils } = require('near-api-js');

// Convert NEAR amount to yoctoNEAR
function toYoctoNEAR(amount) {
  return utils.format.parseNearAmount(amount.toString());
}

// Convert yoctoNEAR to NEAR
function fromYoctoNEAR(amount) {
  return utils.format.formatNearAmount(amount.toString());
}

// Handle errors consistently
function handleError(error, customMessage = '') {
  console.error(customMessage, error);
  return {
    error: error.message || 'An unexpected error occurred',
    details: error.toString()
  };
}

module.exports = {
  toYoctoNEAR,
  fromYoctoNEAR,
  handleError
};