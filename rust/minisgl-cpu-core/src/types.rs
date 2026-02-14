use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub ignore_eos: bool,
    pub max_tokens: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            ignore_eos: false,
            max_tokens: 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Req {
    pub uid: u64,
    pub input_ids: Vec<i32>,
    pub cached_len: usize,
    pub output_len: usize,
    pub sampling_params: SamplingParams,
}

impl Req {
    pub fn new(
        uid: u64,
        input_ids: Vec<i32>,
        cached_len: usize,
        output_len: usize,
        sampling_params: SamplingParams,
    ) -> Self {
        Self {
            uid,
            input_ids,
            cached_len,
            output_len,
            sampling_params,
        }
    }

    pub fn device_len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn remain_len(&self) -> usize {
        self.output_len
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Batch {
    pub reqs: Vec<Req>,
    pub phase: BatchPhase,
}

impl Batch {
    pub fn new(reqs: Vec<Req>, phase: BatchPhase) -> Self {
        Self { reqs, phase }
    }

    pub fn size(&self) -> usize {
        self.reqs.len()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BatchPhase {
    Prefill,
    Decode,
}
