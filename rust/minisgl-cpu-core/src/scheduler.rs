use crate::types::{Batch, BatchPhase, Req};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerPlan {
    pub selected_uids: Vec<u64>,
    pub phase: BatchPhase,
}

impl SchedulerPlan {
    pub fn from_batch(batch: &Batch) -> Self {
        Self {
            selected_uids: batch.reqs.iter().map(|req| req.uid).collect(),
            phase: batch.phase,
        }
    }

    pub fn from_reqs(reqs: &[Req], phase: BatchPhase) -> Self {
        Self {
            selected_uids: reqs.iter().map(|req| req.uid).collect(),
            phase,
        }
    }
}
