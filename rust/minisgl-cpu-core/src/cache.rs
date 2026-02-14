use crate::types::Req;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CacheError {
    #[error("invalid request: input cannot be empty")]
    EmptyInput,
    #[error(
        "invalid cache insert: input and indices lengths differ ({input_len} != {indices_len})"
    )]
    MismatchedInputAndIndices {
        input_len: usize,
        indices_len: usize,
    },
    #[error("cannot evict {requested}, only {evictable} is evictable")]
    EvictTooLarge { requested: usize, evictable: usize },
    #[error("unlock would make node refcount negative")]
    UnlockUnderflow,
    #[error("cache tree is corrupted: {reason}")]
    CorruptedTree { reason: &'static str },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachePrefixMatch {
    pub cached_len: usize,
    pub indices: Vec<i32>,
}

pub trait CacheManager {
    fn match_prefix(&self, req: &Req) -> Result<CachePrefixMatch, CacheError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeInfo {
    pub evictable_size: usize,
    pub protected_size: usize,
}

impl SizeInfo {
    pub fn total_size(self) -> usize {
        self.evictable_size + self.protected_size
    }
}

pub trait PrefixCacheManager {
    type Handle: Clone;

    fn match_prefix(&mut self, input_ids: &[i32]) -> Result<(Self::Handle, Vec<i32>), CacheError>;
    fn lock_handle(&mut self, handle: &Self::Handle, unlock: bool) -> Result<(), CacheError>;
    fn insert_prefix(&mut self, input_ids: &[i32], indices: &[i32]) -> Result<usize, CacheError>;
    fn evict(&mut self, size: usize) -> Result<Vec<i32>, CacheError>;
    fn size_info(&self) -> SizeInfo;
    fn check_integrity(&self) -> Result<(), CacheError>;
}

#[derive(Debug, Default)]
pub struct NoopCacheManager;

impl CacheManager for NoopCacheManager {
    fn match_prefix(&self, req: &Req) -> Result<CachePrefixMatch, CacheError> {
        if req.input_ids.is_empty() {
            return Err(CacheError::EmptyInput);
        }
        Ok(CachePrefixMatch {
            cached_len: 0,
            indices: Vec::new(),
        })
    }
}
