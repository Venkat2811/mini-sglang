use crate::types::Req;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CacheError {
    #[error("invalid request: input cannot be empty")]
    EmptyInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachePrefixMatch {
    pub cached_len: usize,
    pub indices: Vec<i32>,
}

pub trait CacheManager {
    fn match_prefix(&self, req: &Req) -> Result<CachePrefixMatch, CacheError>;
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
