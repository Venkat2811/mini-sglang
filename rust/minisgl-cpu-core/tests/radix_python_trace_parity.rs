use std::{collections::HashMap, fs, path::PathBuf};

use minisgl_cpu_core::{PrefixCacheManager, RadixCacheHandle, RadixCacheManager};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TracePayload {
    cases: Vec<TraceCase>,
}

#[derive(Debug, Deserialize)]
struct TraceCase {
    name: String,
    ops: Vec<TraceOp>,
}

#[derive(Debug, Deserialize)]
struct TraceSize {
    evictable_size: usize,
    protected_size: usize,
}

#[derive(Debug, Deserialize)]
struct TraceOp {
    op: String,
    input_ids: Option<Vec<i32>>,
    indices: Option<Vec<i32>>,
    slot: Option<String>,
    unlock: Option<bool>,
    size: Option<usize>,
    expect_prefix_len: Option<usize>,
    expect_cached_len: Option<usize>,
    expect_indices: Option<Vec<i32>>,
    expect_evicted: Option<Vec<i32>>,
    expect_size: TraceSize,
}

fn trace_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/radix_golden_trace.yaml")
}

#[test]
fn replay_python_golden_traces() {
    let text = fs::read_to_string(trace_path()).expect("read golden trace payload");
    let payload: TracePayload = serde_json::from_str(&text).expect("parse golden trace payload");

    for case in payload.cases {
        let mut manager = RadixCacheManager::new();
        let mut handles: HashMap<String, RadixCacheHandle> = HashMap::new();

        for (idx, op) in case.ops.iter().enumerate() {
            match op.op.as_str() {
                "insert" => {
                    let input_ids = op.input_ids.as_ref().expect("insert.input_ids");
                    let indices = op.indices.as_ref().expect("insert.indices");
                    let prefix_len = manager
                        .insert_prefix(input_ids, indices)
                        .expect("insert_prefix must succeed");
                    assert_eq!(
                        Some(prefix_len),
                        op.expect_prefix_len,
                        "case={} op#{} insert prefix_len mismatch",
                        case.name,
                        idx
                    );
                }
                "match" => {
                    let input_ids = op.input_ids.as_ref().expect("match.input_ids");
                    let slot = op.slot.as_ref().expect("match.slot").clone();
                    let (handle, matched) = manager
                        .match_prefix(input_ids)
                        .expect("match_prefix must succeed");
                    assert_eq!(
                        Some(handle.cached_len),
                        op.expect_cached_len,
                        "case={} op#{} cached_len mismatch",
                        case.name,
                        idx
                    );
                    assert_eq!(
                        Some(matched),
                        op.expect_indices.clone(),
                        "case={} op#{} matched indices mismatch",
                        case.name,
                        idx
                    );
                    handles.insert(slot, handle);
                }
                "lock" => {
                    let slot = op.slot.as_ref().expect("lock.slot");
                    let unlock = op.unlock.expect("lock.unlock");
                    let handle = handles.get(slot).expect("lock handle slot exists");
                    manager
                        .lock_handle(handle, unlock)
                        .expect("lock_handle must succeed");
                }
                "evict" => {
                    let size = op.size.expect("evict.size");
                    let evicted = manager.evict(size).expect("evict must succeed");
                    assert_eq!(
                        Some(evicted),
                        op.expect_evicted.clone(),
                        "case={} op#{} evicted indices mismatch",
                        case.name,
                        idx
                    );
                }
                other => panic!("unsupported op '{}'", other),
            }

            let size_info = manager.size_info();
            assert_eq!(
                size_info.evictable_size, op.expect_size.evictable_size,
                "case={} op#{} evictable size mismatch",
                case.name, idx
            );
            assert_eq!(
                size_info.protected_size, op.expect_size.protected_size,
                "case={} op#{} protected size mismatch",
                case.name, idx
            );
            manager
                .check_integrity()
                .expect("integrity should hold after each operation");
        }
    }
}
