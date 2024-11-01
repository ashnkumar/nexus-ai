use calimero_sdk::{
  app,
  borsh::{BorshDeserialize, BorshSerialize},
  env,
};

#[app::event]
pub enum Event {
  ModelUpdateSubmitted {
      trader_id: String,
      update_id: String,
      timestamp: u64,
      training_metrics: TrainingMetrics,
  },
  AggregationStarted {
      round: u64,
      timestamp: u64,
  },
  AggregationCompleted {
      round: u64,
      global_model_id: String,
      timestamp: u64,
  },
  ValidationMetricsUpdated {
      round: u64,
      metrics: ValidationMetrics,
      timestamp: u64,
  },
}

#[derive(BorshSerialize, BorshDeserialize, Clone)]
pub struct TrainingMetrics {
  pub loss: f64,
  pub accuracy: f64,
  pub epochs_completed: u32,
}

#[derive(BorshSerialize, BorshDeserialize, Clone)]
pub struct ValidationMetrics {
  pub accuracy: f64,
  pub precision: f64,
  pub recall: f64,
  pub f1_score: f64,
}

#[derive(BorshSerialize, BorshDeserialize, Clone)]
pub struct ModelUpdate {
  pub trader_id: String,
  pub update_id: String,
  pub timestamp: u64,
  pub metrics: TrainingMetrics,
}

#[app::state(emits = Event)]
#[derive(Default, BorshDeserialize, BorshSerialize)]
#[borsh(crate = "calimero_sdk::borsh")]
pub struct FederatedLearning {
  // Training state
  current_round: u64,
  aggregation_in_progress: bool,
  minimum_traders_for_aggregation: u32,
  
  // Model tracking
  latest_global_model_id: Option<String>,
  trader_updates: Vec<ModelUpdate>,
  
  // Metrics
  current_validation_metrics: Option<ValidationMetrics>,
  
  // Configuration
  owner: String,
  approved_traders: Vec<String>,
}

#[app::logic]
impl FederatedLearning {
  #[app::init]
  pub fn init(owner: String, minimum_traders: u32) -> Self {
      Self {
          current_round: 0,
          aggregation_in_progress: false,
          minimum_traders_for_aggregation: minimum_traders,
          latest_global_model_id: None,
          trader_updates: Vec::new(),
          current_validation_metrics: None,
          owner,
          approved_traders: Vec::new(),
      }
  }

  // Admin functions
  pub fn add_approved_trader(&mut self, trader_id: String) {
      assert_eq!(env::predecessor_account_id(), self.owner, "Unauthorized");
      if !self.approved_traders.contains(&trader_id) {
          self.approved_traders.push(trader_id);
      }
  }

  pub fn remove_approved_trader(&mut self, trader_id: &str) {
      assert_eq!(env::predecessor_account_id(), self.owner, "Unauthorized");
      self.approved_traders.retain(|x| x != trader_id);
  }

  // Model update functions
  pub fn register_model_update(
      &mut self,
      trader_id: String,
      update_id: String,
      metrics: TrainingMetrics,
  ) {
      // Validate trader is approved
      assert!(
          self.approved_traders.contains(&trader_id),
          "Trader not approved"
      );
      
      // Validate not in aggregation
      assert!(!self.aggregation_in_progress, "Aggregation in progress");

      let timestamp = env::block_timestamp();
      
      // Add update
      let update = ModelUpdate {
          trader_id: trader_id.clone(),
          update_id: update_id.clone(),
          timestamp,
          metrics: metrics.clone(),
      };
      
      self.trader_updates.push(update);

      // Emit event
      app::emit!(Event::ModelUpdateSubmitted {
          trader_id,
          update_id,
          timestamp,
          training_metrics: metrics,
      });
  }

  pub fn start_aggregation(&mut self) {
      // Only owner can start aggregation
      assert_eq!(env::predecessor_account_id(), self.owner, "Unauthorized");
      
      // Verify enough updates
      assert!(
          self.trader_updates.len() >= self.minimum_traders_for_aggregation as usize,
          "Not enough trader updates"
      );
      
      // Verify not already in progress
      assert!(!self.aggregation_in_progress, "Aggregation already in progress");

      self.aggregation_in_progress = true;
      self.current_round += 1;

      app::emit!(Event::AggregationStarted {
          round: self.current_round,
          timestamp: env::block_timestamp(),
      });
  }

  pub fn finish_aggregation(
      &mut self,
      global_model_id: String,
      validation_metrics: ValidationMetrics,
  ) {
      // Validate caller and state
      assert_eq!(env::predecessor_account_id(), self.owner, "Unauthorized");
      assert!(self.aggregation_in_progress, "No aggregation in progress");

      // Update state
      self.latest_global_model_id = Some(global_model_id.clone());
      self.current_validation_metrics = Some(validation_metrics.clone());
      self.aggregation_in_progress = false;
      self.trader_updates.clear();

      // Emit events
      app::emit!(Event::AggregationCompleted {
          round: self.current_round,
          global_model_id,
          timestamp: env::block_timestamp(),
      });

      app::emit!(Event::ValidationMetricsUpdated {
          round: self.current_round,
          metrics: validation_metrics,
          timestamp: env::block_timestamp(),
      });
  }

  // View functions
  pub fn get_latest_model_id(&self) -> Option<String> {
      self.latest_global_model_id.clone()
  }

  pub fn get_pending_updates(&self) -> Vec<ModelUpdate> {
      self.trader_updates.clone()
  }

  pub fn get_current_round(&self) -> u64 {
      self.current_round
  }

  pub fn get_validation_metrics(&self) -> Option<ValidationMetrics> {
      self.current_validation_metrics.clone()
  }

  pub fn is_trader_approved(&self, trader_id: &str) -> bool {
      self.approved_traders.contains(&trader_id.to_string())
  }

  pub fn get_minimum_traders(&self) -> u32 {
      self.minimum_traders_for_aggregation
  }
}