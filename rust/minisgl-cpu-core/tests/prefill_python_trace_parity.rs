use std::{fs, path::PathBuf};

use minisgl_cpu_core::{
    make_input_tuple, make_positions, make_write_tuple, CacheMatch, PendingReq, PrefillAdder,
    PrefillCache, PrefillTable, ScheduledReq,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct PrefillTracePayload {
    adder_cases: Vec<AdderCase>,
    mapping_case: MappingCase,
    decode_only_case: DecodeOnlyCase,
}

#[derive(Debug, Deserialize)]
struct AdderCase {
    name: String,
    token_budget: usize,
    reserved_size: usize,
    cache_available_size: usize,
    table_slots: Vec<i32>,
    cached_len: usize,
    match_indices: Vec<i32>,
    lock_impact: usize,
    input_ids: Vec<i32>,
    output_len: usize,
    expected: AdderExpected,
}

#[derive(Debug, Deserialize)]
struct AdderExpected {
    admitted: bool,
    token_budget_after: usize,
    reserved_size_after: usize,
    is_chunked: Option<bool>,
    cached_len: Option<usize>,
    device_len: Option<usize>,
    table_idx: Option<i32>,
    extend_len: Option<usize>,
    remain_len: Option<usize>,
    can_decode: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct MappingCase {
    positions: Vec<i32>,
    input_mapping: Vec<i32>,
    input_positions: Vec<i32>,
    write_req_mapping: Vec<i32>,
    write_pos: Vec<i32>,
}

#[derive(Debug, Deserialize)]
struct DecodeOnlyCase {
    write_req_mapping: Vec<i32>,
    write_pos: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Handle {
    id: u64,
}

#[derive(Debug)]
struct FakeCache {
    available_size: usize,
    cached_len: usize,
    match_indices: Vec<i32>,
    lock_impact: usize,
}

impl PrefillCache for FakeCache {
    type Handle = Handle;

    fn match_req(
        &mut self,
        _input_ids_without_last: &[i32],
    ) -> Result<CacheMatch<Self::Handle>, String> {
        Ok(CacheMatch {
            handle: Handle { id: 1 },
            cached_len: self.cached_len,
            match_indices: self.match_indices.clone(),
        })
    }

    fn lock(&mut self, _handle: &Self::Handle) -> Result<(), String> {
        self.available_size = self.available_size.saturating_sub(self.lock_impact);
        Ok(())
    }

    fn unlock(&mut self, _handle: &Self::Handle) -> Result<(), String> {
        self.available_size += self.lock_impact;
        Ok(())
    }

    fn available_size(&self) -> usize {
        self.available_size
    }
}

#[derive(Debug)]
struct FakeTable {
    slots: Vec<i32>,
}

impl PrefillTable for FakeTable {
    fn available_size(&self) -> usize {
        self.slots.len()
    }

    fn allocate(&mut self) -> Option<i32> {
        self.slots.pop()
    }
}

fn trace_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/prefill_golden_trace.yaml")
}

#[test]
fn replay_python_prefill_adder_cases() {
    let text = fs::read_to_string(trace_path()).expect("read prefill golden trace");
    let payload: PrefillTracePayload = serde_json::from_str(&text).expect("parse prefill trace");

    for case in payload.adder_cases {
        let cache = FakeCache {
            available_size: case.cache_available_size,
            cached_len: case.cached_len,
            match_indices: case.match_indices,
            lock_impact: case.lock_impact,
        };
        let mut slots = case.table_slots;
        slots.reverse();
        let table = FakeTable { slots };
        let mut cache = cache;
        let mut table = table;

        let pending = PendingReq {
            uid: 999,
            input_ids: case.input_ids,
            output_len: case.output_len,
            chunked_req: None,
        };
        let mut adder = PrefillAdder {
            token_budget: case.token_budget,
            reserved_size: case.reserved_size,
            cache: &mut cache,
            table: &mut table,
        };

        let result = adder
            .try_add_one(&pending)
            .expect("adder case should not error");
        assert_eq!(
            result.is_some(),
            case.expected.admitted,
            "adder case '{}' admitted mismatch",
            case.name
        );
        assert_eq!(
            adder.token_budget, case.expected.token_budget_after,
            "adder case '{}' token_budget_after mismatch",
            case.name
        );
        assert_eq!(
            adder.reserved_size, case.expected.reserved_size_after,
            "adder case '{}' reserved_size_after mismatch",
            case.name
        );

        if let Some(req) = result {
            assert_eq!(Some(req.is_chunked), case.expected.is_chunked);
            assert_eq!(Some(req.cached_len), case.expected.cached_len);
            assert_eq!(Some(req.device_len), case.expected.device_len);
            assert_eq!(Some(req.table_idx), case.expected.table_idx);
            assert_eq!(Some(req.extend_len()), case.expected.extend_len);
            assert_eq!(Some(req.remain_len()), case.expected.remain_len);
            assert_eq!(Some(req.can_decode()), case.expected.can_decode);
        }
    }
}

#[test]
fn replay_python_mapping_case() {
    let text = fs::read_to_string(trace_path()).expect("read prefill golden trace");
    let payload: PrefillTracePayload = serde_json::from_str(&text).expect("parse prefill trace");
    let m = payload.mapping_case;

    let reqs = vec![
        ScheduledReq {
            uid: 1,
            table_idx: 7,
            cached_len: 2,
            device_len: 5,
            max_device_len: 9,
            output_len: 4,
            cache_handle: Handle { id: 11 },
            is_chunked: false,
        },
        ScheduledReq {
            uid: 2,
            table_idx: 9,
            cached_len: 1,
            device_len: 3,
            max_device_len: 10,
            output_len: 7,
            cache_handle: Handle { id: 22 },
            is_chunked: true,
        },
        ScheduledReq {
            uid: 3,
            table_idx: 11,
            cached_len: 4,
            device_len: 5,
            max_device_len: 6,
            output_len: 1,
            cache_handle: Handle { id: 33 },
            is_chunked: false,
        },
    ];

    let positions = make_positions(&reqs);
    let (input_mapping, input_positions) = make_input_tuple(&reqs, &positions);
    let (write_req_mapping, write_pos) = make_write_tuple(&reqs);

    assert_eq!(positions, m.positions);
    assert_eq!(input_mapping, m.input_mapping);
    assert_eq!(input_positions, m.input_positions);
    assert_eq!(write_req_mapping, m.write_req_mapping);
    assert_eq!(write_pos, m.write_pos);

    let decode = payload.decode_only_case;
    let decode_reqs = vec![
        ScheduledReq {
            uid: 10,
            table_idx: 13,
            cached_len: 5,
            device_len: 6,
            max_device_len: 7,
            output_len: 1,
            cache_handle: Handle { id: 101 },
            is_chunked: false,
        },
        ScheduledReq {
            uid: 11,
            table_idx: 14,
            cached_len: 3,
            device_len: 4,
            max_device_len: 6,
            output_len: 2,
            cache_handle: Handle { id: 102 },
            is_chunked: false,
        },
    ];
    let (decode_req_mapping, decode_write_pos) = make_write_tuple(&decode_reqs);
    assert_eq!(decode_req_mapping, decode.write_req_mapping);
    assert_eq!(decode_write_pos, decode.write_pos);
}
