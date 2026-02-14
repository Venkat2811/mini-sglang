use std::time::Instant;

use minisgl_cpu_core::{make_input_tuple, make_positions, make_write_tuple, ScheduledReq};

#[derive(Clone, Debug)]
struct DummyHandle;

fn main() {
    let batch_size = 128usize;
    let extend_len = 32usize;
    let mut reqs = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let cached_len = 64usize;
        let device_len = cached_len + extend_len;
        reqs.push(ScheduledReq {
            uid: i as u64,
            table_idx: i as i32,
            cached_len,
            device_len,
            max_device_len: device_len + 64,
            output_len: 64,
            cache_handle: DummyHandle,
            is_chunked: false,
        });
    }

    let warmup_iters = 5_000usize;
    for _ in 0..warmup_iters {
        let positions = make_positions(&reqs);
        let _ = make_input_tuple(&reqs, &positions);
        let _ = make_write_tuple(&reqs);
    }

    let iters = 20_000usize;
    let start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iters {
        let positions = make_positions(&reqs);
        let (input_mapping, input_positions) = make_input_tuple(&reqs, &positions);
        let (write_mapping, write_pos) = make_write_tuple(&reqs);
        checksum += positions.len();
        checksum += input_mapping.len();
        checksum += input_positions.len();
        checksum += write_mapping.len();
        checksum += write_pos.len();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let ops_per_sec = iters as f64 / elapsed;

    println!("prefill_mapping_ops_per_sec={ops_per_sec:.2}");
    println!("prefill_mapping_checksum={checksum}");
}
