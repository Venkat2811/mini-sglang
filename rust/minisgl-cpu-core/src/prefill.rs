use std::collections::VecDeque;

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PrefillError {
    #[error("input length must be > 0")]
    EmptyInput,
    #[error("table allocation failed: no free slots")]
    TableExhausted,
    #[error("cache backend error: {0}")]
    CacheBackend(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingReq<H> {
    pub uid: u64,
    pub input_ids: Vec<i32>,
    pub output_len: usize,
    pub chunked_req: Option<ChunkedReqState<H>>,
}

impl<H> PendingReq<H> {
    pub fn input_len(&self) -> usize {
        self.input_ids.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkedReqState<H> {
    pub cache_handle: H,
    pub table_idx: i32,
    pub cached_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledReq<H> {
    pub uid: u64,
    pub table_idx: i32,
    pub cached_len: usize,
    pub device_len: usize,
    pub max_device_len: usize,
    pub output_len: usize,
    pub cache_handle: H,
    pub is_chunked: bool,
}

impl<H> ScheduledReq<H> {
    pub fn remain_len(&self) -> usize {
        self.max_device_len.saturating_sub(self.device_len)
    }

    pub fn extend_len(&self) -> usize {
        self.device_len.saturating_sub(self.cached_len)
    }

    pub fn can_decode(&self) -> bool {
        !self.is_chunked && self.output_len > 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheMatch<H> {
    pub handle: H,
    pub cached_len: usize,
    pub match_indices: Vec<i32>,
}

pub trait PrefillCache {
    type Handle: Clone;

    fn match_req(
        &mut self,
        input_ids_without_last: &[i32],
    ) -> Result<CacheMatch<Self::Handle>, String>;
    fn lock(&mut self, handle: &Self::Handle) -> Result<(), String>;
    fn unlock(&mut self, handle: &Self::Handle) -> Result<(), String>;
    fn available_size(&self) -> usize;
}

pub trait PrefillTable {
    fn available_size(&self) -> usize;
    fn allocate(&mut self) -> Option<i32>;
}

#[derive(Debug)]
pub struct PrefillAdder<'a, C, T>
where
    C: PrefillCache,
    T: PrefillTable,
{
    pub token_budget: usize,
    pub reserved_size: usize,
    pub cache: &'a mut C,
    pub table: &'a mut T,
}

impl<'a, C, T> PrefillAdder<'a, C, T>
where
    C: PrefillCache,
    T: PrefillTable,
{
    fn try_allocate_one(
        &mut self,
        req: &PendingReq<C::Handle>,
    ) -> Result<Option<ChunkedReqState<C::Handle>>, PrefillError> {
        if self.table.available_size() == 0 {
            return Ok(None);
        }

        if req.input_len() == 0 {
            return Err(PrefillError::EmptyInput);
        }

        let match_input = &req.input_ids[..req.input_len() - 1];
        let matched = self
            .cache
            .match_req(match_input)
            .map_err(PrefillError::CacheBackend)?;
        let cached_len = matched.cached_len;

        let extend_len = req.input_len().saturating_sub(cached_len);
        let estimated_len = extend_len + req.output_len;
        if estimated_len + self.reserved_size > self.cache.available_size() {
            return Ok(None);
        }

        self.cache
            .lock(&matched.handle)
            .map_err(PrefillError::CacheBackend)?;
        if estimated_len + self.reserved_size > self.cache.available_size() {
            self.cache
                .unlock(&matched.handle)
                .map_err(PrefillError::CacheBackend)?;
            return Ok(None);
        }

        let table_idx = self.table.allocate().ok_or(PrefillError::TableExhausted)?;
        Ok(Some(ChunkedReqState {
            cache_handle: matched.handle,
            table_idx,
            cached_len,
        }))
    }

    fn add_one_req(
        &mut self,
        pending_req: &PendingReq<C::Handle>,
        allocated: ChunkedReqState<C::Handle>,
    ) -> ScheduledReq<C::Handle> {
        let remain_len = pending_req.input_len().saturating_sub(allocated.cached_len);
        let chunk_size = self.token_budget.min(remain_len);
        let is_chunked = chunk_size < remain_len;

        self.token_budget = self.token_budget.saturating_sub(chunk_size);
        self.reserved_size += remain_len + pending_req.output_len;

        let device_len = allocated.cached_len + chunk_size;
        ScheduledReq {
            uid: pending_req.uid,
            table_idx: allocated.table_idx,
            cached_len: allocated.cached_len,
            device_len,
            max_device_len: device_len + pending_req.output_len,
            output_len: pending_req.output_len,
            cache_handle: allocated.cache_handle,
            is_chunked,
        }
    }

    pub fn try_add_one(
        &mut self,
        pending_req: &PendingReq<C::Handle>,
    ) -> Result<Option<ScheduledReq<C::Handle>>, PrefillError> {
        if self.token_budget == 0 {
            return Ok(None);
        }

        if let Some(chunked) = pending_req.chunked_req.clone() {
            return Ok(Some(self.add_one_req(pending_req, chunked)));
        }

        if let Some(allocated) = self.try_allocate_one(pending_req)? {
            return Ok(Some(self.add_one_req(pending_req, allocated)));
        }

        Ok(None)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillBatch<H> {
    pub reqs: Vec<ScheduledReq<H>>,
}

#[derive(Debug)]
pub struct PrefillManager<C, T>
where
    C: PrefillCache,
    T: PrefillTable,
{
    pub cache: C,
    pub table: T,
    pub pending: VecDeque<PendingReq<C::Handle>>,
}

impl<C, T> PrefillManager<C, T>
where
    C: PrefillCache,
    T: PrefillTable,
{
    pub fn new(cache: C, table: T) -> Self {
        Self {
            cache,
            table,
            pending: VecDeque::new(),
        }
    }

    pub fn add_pending(&mut self, req: PendingReq<C::Handle>) {
        self.pending.push_back(req);
    }

    pub fn schedule_next_batch(
        &mut self,
        prefill_budget: usize,
        decode_inflight_tokens: usize,
    ) -> Result<Option<PrefillBatch<C::Handle>>, PrefillError> {
        if self.pending.is_empty() {
            return Ok(None);
        }

        let mut adder = PrefillAdder {
            token_budget: prefill_budget,
            reserved_size: decode_inflight_tokens,
            cache: &mut self.cache,
            table: &mut self.table,
        };
        let mut reqs = Vec::<ScheduledReq<C::Handle>>::new();
        let mut chunked = VecDeque::<PendingReq<C::Handle>>::new();
        let mut consumed = 0usize;

        for pending_req in self.pending.iter() {
            if let Some(req) = adder.try_add_one(pending_req)? {
                let mut next_pending = pending_req.clone();
                next_pending.chunked_req = None;
                if req.is_chunked {
                    next_pending.chunked_req = Some(ChunkedReqState {
                        cache_handle: req.cache_handle.clone(),
                        table_idx: req.table_idx,
                        cached_len: req.device_len,
                    });
                    chunked.push_back(next_pending);
                }
                reqs.push(req);
                consumed += 1;
            } else {
                break;
            }
        }

        if reqs.is_empty() {
            return Ok(None);
        }

        for _ in 0..consumed {
            let _ = self.pending.pop_front();
        }
        while let Some(req) = chunked.pop_back() {
            self.pending.push_front(req);
        }

        Ok(Some(PrefillBatch { reqs }))
    }
}

pub fn decode_inflight_tokens<H>(running_reqs: &[ScheduledReq<H>]) -> usize {
    running_reqs
        .iter()
        .filter(|req| req.can_decode())
        .map(ScheduledReq::remain_len)
        .sum()
}

pub fn make_positions<H>(padded_reqs: &[ScheduledReq<H>]) -> Vec<i32> {
    let total: usize = padded_reqs.iter().map(ScheduledReq::extend_len).sum();
    let mut out = Vec::with_capacity(total);
    for req in padded_reqs {
        for pos in req.cached_len..req.device_len {
            out.push(pos as i32);
        }
    }
    out
}

pub fn make_input_mapping<H>(padded_reqs: &[ScheduledReq<H>]) -> Vec<i32> {
    let total: usize = padded_reqs.iter().map(ScheduledReq::extend_len).sum();
    let mut mapping = Vec::with_capacity(total);
    for req in padded_reqs {
        for _ in 0..req.extend_len() {
            mapping.push(req.table_idx);
        }
    }
    mapping
}

pub fn make_input_tuple<H>(
    padded_reqs: &[ScheduledReq<H>],
    positions: &[i32],
) -> (Vec<i32>, Vec<i32>) {
    (make_input_mapping(padded_reqs), positions.to_vec())
}

pub fn make_write_tuple<H>(reqs: &[ScheduledReq<H>]) -> (Vec<i32>, Vec<i32>) {
    let req_mapping: Vec<i32> = reqs.iter().map(|req| req.table_idx).collect();
    let write_mapping: Vec<i32> = reqs
        .iter()
        .map(|req| {
            if req.can_decode() {
                req.device_len as i32
            } else {
                -1
            }
        })
        .collect();
    (req_mapping, write_mapping)
}
