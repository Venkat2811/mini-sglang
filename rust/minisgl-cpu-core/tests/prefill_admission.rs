use std::collections::HashMap;

use minisgl_cpu_core::{
    decode_inflight_tokens, make_input_tuple, make_positions, make_write_tuple, CacheMatch,
    PendingReq, PrefillAdder, PrefillCache, PrefillManager, PrefillTable, ScheduledReq,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FakeHandle {
    id: u64,
}

#[derive(Debug)]
struct FakeCache {
    available_size: usize,
    matches: HashMap<Vec<i32>, CacheMatch<FakeHandle>>,
    lock_impact: HashMap<u64, usize>,
}

impl FakeCache {
    fn new(available_size: usize) -> Self {
        Self {
            available_size,
            matches: HashMap::new(),
            lock_impact: HashMap::new(),
        }
    }

    fn with_match(mut self, key: Vec<i32>, handle_id: u64, cached_len: usize, indices: Vec<i32>) -> Self {
        self.matches.insert(
            key,
            CacheMatch {
                handle: FakeHandle { id: handle_id },
                cached_len,
                match_indices: indices,
            },
        );
        self
    }

    fn with_lock_impact(mut self, handle_id: u64, impact: usize) -> Self {
        self.lock_impact.insert(handle_id, impact);
        self
    }
}

impl PrefillCache for FakeCache {
    type Handle = FakeHandle;

    fn match_req(&mut self, input_ids_without_last: &[i32]) -> Result<CacheMatch<Self::Handle>, String> {
        self.matches
            .get(input_ids_without_last)
            .cloned()
            .ok_or_else(|| format!("no fake match for key {input_ids_without_last:?}"))
    }

    fn lock(&mut self, handle: &Self::Handle) -> Result<(), String> {
        let impact = *self.lock_impact.get(&handle.id).unwrap_or(&0);
        self.available_size = self.available_size.saturating_sub(impact);
        Ok(())
    }

    fn unlock(&mut self, handle: &Self::Handle) -> Result<(), String> {
        let impact = *self.lock_impact.get(&handle.id).unwrap_or(&0);
        self.available_size += impact;
        Ok(())
    }

    fn available_size(&self) -> usize {
        self.available_size
    }
}

#[derive(Debug)]
struct FakeTable {
    free_slots: Vec<i32>,
}

impl FakeTable {
    fn new(mut free_slots: Vec<i32>) -> Self {
        free_slots.reverse();
        Self { free_slots }
    }
}

impl PrefillTable for FakeTable {
    fn available_size(&self) -> usize {
        self.free_slots.len()
    }

    fn allocate(&mut self) -> Option<i32> {
        self.free_slots.pop()
    }
}

fn pending(uid: u64, ids: &[i32], output_len: usize) -> PendingReq<FakeHandle> {
    PendingReq {
        uid,
        input_ids: ids.to_vec(),
        output_len,
        chunked_req: None,
    }
}

#[test]
fn adder_rejects_near_capacity() {
    let mut cache = FakeCache::new(10).with_match(vec![1, 2, 3, 4], 1, 1, vec![9]);
    let mut table = FakeTable::new(vec![4]);
    let req = pending(7, &[1, 2, 3, 4, 5], 5);
    let mut adder = PrefillAdder {
        token_budget: 16,
        reserved_size: 2,
        cache: &mut cache,
        table: &mut table,
    };

    let scheduled = adder.try_add_one(&req).expect("adder should not error");
    assert!(scheduled.is_none());
}

#[test]
fn adder_chunks_when_token_budget_is_small() {
    let mut cache = FakeCache::new(64).with_match(vec![10, 11, 12, 13], 8, 1, vec![3]);
    let mut table = FakeTable::new(vec![2]);
    let req = pending(42, &[10, 11, 12, 13, 14], 3);
    let mut adder = PrefillAdder {
        token_budget: 2,
        reserved_size: 0,
        cache: &mut cache,
        table: &mut table,
    };

    let scheduled = adder
        .try_add_one(&req)
        .expect("adder should not error")
        .expect("request should be admitted");
    assert!(scheduled.is_chunked);
    assert_eq!(scheduled.cached_len, 1);
    assert_eq!(scheduled.device_len, 3);
    assert_eq!(scheduled.extend_len(), 2);
    assert_eq!(adder.token_budget, 0);
    assert_eq!(adder.reserved_size, 7);
}

#[test]
fn manager_requeues_chunked_and_respects_inflight_budget() {
    let cache = FakeCache::new(64)
        .with_match(vec![1, 2, 3, 4, 5], 1, 0, vec![])
        .with_match(vec![9, 8, 7], 2, 0, vec![])
        .with_lock_impact(1, 0)
        .with_lock_impact(2, 0);
    let table = FakeTable::new(vec![10, 11, 12]);
    let mut manager = PrefillManager::new(cache, table);

    manager.add_pending(pending(100, &[1, 2, 3, 4, 5, 6], 2));
    manager.add_pending(pending(200, &[9, 8, 7, 6], 2));

    let first = manager
        .schedule_next_batch(3, 4)
        .expect("first schedule should succeed")
        .expect("first batch should exist");
    assert_eq!(first.reqs.len(), 1);
    assert!(first.reqs[0].is_chunked);
    assert_eq!(first.reqs[0].uid, 100);
    assert_eq!(manager.pending.len(), 2);
    assert_eq!(manager.pending[0].uid, 100);
    assert!(manager.pending[0].chunked_req.is_some());

    let second = manager
        .schedule_next_batch(3, 0)
        .expect("second schedule should succeed")
        .expect("second batch should exist");
    assert_eq!(second.reqs.len(), 1);
    assert_eq!(second.reqs[0].uid, 100);
    assert!(!second.reqs[0].is_chunked);
    assert_eq!(manager.pending.len(), 1);
    assert_eq!(manager.pending[0].uid, 200);
}

#[test]
fn mapping_builders_match_python_contracts_for_mixed_batch() {
    let req_a = ScheduledReq {
        uid: 1,
        table_idx: 7,
        cached_len: 2,
        device_len: 5,
        max_device_len: 9,
        output_len: 4,
        cache_handle: FakeHandle { id: 1 },
        is_chunked: false,
    };
    let req_b = ScheduledReq {
        uid: 2,
        table_idx: 9,
        cached_len: 1,
        device_len: 3,
        max_device_len: 10,
        output_len: 7,
        cache_handle: FakeHandle { id: 2 },
        is_chunked: true,
    };
    let req_c = ScheduledReq {
        uid: 3,
        table_idx: 11,
        cached_len: 4,
        device_len: 5,
        max_device_len: 6,
        output_len: 1,
        cache_handle: FakeHandle { id: 3 },
        is_chunked: false,
    };

    let padded = vec![req_a.clone(), req_b.clone(), req_c.clone()];
    let positions = make_positions(&padded);
    assert_eq!(positions, vec![2, 3, 4, 1, 2, 4]);

    let (input_mapping, input_positions) = make_input_tuple(&padded, &positions);
    assert_eq!(input_mapping, vec![7, 7, 7, 9, 9, 11]);
    assert_eq!(input_positions, positions);

    let (write_req_mapping, write_pos) = make_write_tuple(&[req_a.clone(), req_b.clone(), req_c.clone()]);
    assert_eq!(write_req_mapping, vec![7, 9, 11]);
    assert_eq!(write_pos, vec![5, -1, 5]);

    let inflight = decode_inflight_tokens(&[req_a, req_b, req_c]);
    assert_eq!(inflight, 5);
}
