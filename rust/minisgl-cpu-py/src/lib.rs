use std::collections::HashMap;

use minisgl_cpu_core::{
    make_input_mapping as core_make_input_mapping, make_positions as core_make_positions,
    make_write_tuple as core_make_write_tuple, CacheMatch, PendingReq, PrefillAdder, PrefillCache,
    PrefillTable, PrefixCacheManager, RadixCacheHandle, RadixCacheManager, SamplingParams,
    ScheduledReq,
};
use pyo3::{
    exceptions::{PyKeyError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyByteArray,
};

#[pyclass(name = "SamplingParams")]
#[derive(Clone, Debug)]
struct PySamplingParams {
    #[pyo3(get, set)]
    temperature: f32,
    #[pyo3(get, set)]
    top_k: i32,
    #[pyo3(get, set)]
    top_p: f32,
    #[pyo3(get, set)]
    ignore_eos: bool,
    #[pyo3(get, set)]
    max_tokens: u32,
}

#[pymethods]
impl PySamplingParams {
    #[new]
    fn new() -> Self {
        let p = SamplingParams::default();
        Self {
            temperature: p.temperature,
            top_k: p.top_k,
            top_p: p.top_p,
            ignore_eos: p.ignore_eos,
            max_tokens: p.max_tokens,
        }
    }
}

fn cache_err(err: impl ToString) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

#[derive(Clone, Debug)]
struct DummyHandle;

fn make_reqs_for_positions(
    cached_lens: &[usize],
    device_lens: &[usize],
) -> PyResult<Vec<ScheduledReq<DummyHandle>>> {
    if cached_lens.len() != device_lens.len() {
        return Err(PyValueError::new_err(
            "cached_lens and device_lens lengths must match",
        ));
    }
    let mut reqs = Vec::with_capacity(cached_lens.len());
    for (idx, (&cached_len, &device_len)) in cached_lens.iter().zip(device_lens).enumerate() {
        reqs.push(ScheduledReq {
            uid: idx as u64,
            table_idx: idx as i32,
            cached_len,
            device_len,
            max_device_len: device_len + 1,
            output_len: 1,
            cache_handle: DummyHandle,
            is_chunked: false,
        });
    }
    Ok(reqs)
}

#[pyclass(name = "RadixCacheManager", unsendable)]
struct PyRadixCacheManager {
    inner: RadixCacheManager,
    handles: HashMap<u64, RadixCacheHandle>,
    next_handle_id: u64,
}

#[pymethods]
impl PyRadixCacheManager {
    #[new]
    fn new() -> Self {
        Self {
            inner: RadixCacheManager::new(),
            handles: HashMap::new(),
            next_handle_id: 1,
        }
    }

    fn insert_prefix(&mut self, input_ids: Vec<i32>, indices: Vec<i32>) -> PyResult<usize> {
        self.inner
            .insert_prefix(&input_ids, &indices)
            .map_err(cache_err)
    }

    fn match_prefix(&mut self, input_ids: Vec<i32>) -> PyResult<(u64, usize, Vec<i32>)> {
        let (handle, indices) = self.inner.match_prefix(&input_ids).map_err(cache_err)?;
        let handle_id = self.next_handle_id;
        self.next_handle_id += 1;
        let cached_len = handle.cached_len;
        self.handles.insert(handle_id, handle);
        Ok((handle_id, cached_len, indices))
    }

    #[pyo3(signature = (handle_id, unlock=false))]
    fn lock_handle(&mut self, handle_id: u64, unlock: bool) -> PyResult<()> {
        let handle = self
            .handles
            .get(&handle_id)
            .ok_or_else(|| PyKeyError::new_err(format!("unknown handle_id={handle_id}")))?
            .clone();
        self.inner.lock_handle(&handle, unlock).map_err(cache_err)
    }

    fn evict(&mut self, size: usize) -> PyResult<Vec<i32>> {
        self.inner.evict(size).map_err(cache_err)
    }

    fn size_info(&self) -> (usize, usize) {
        let size = self.inner.size_info();
        (size.evictable_size, size.protected_size)
    }

    fn check_integrity(&self) -> PyResult<()> {
        self.inner.check_integrity().map_err(cache_err)
    }
}

#[pyfunction]
fn make_positions(cached_lens: Vec<usize>, device_lens: Vec<usize>) -> PyResult<Vec<i32>> {
    let reqs = make_reqs_for_positions(&cached_lens, &device_lens)?;
    Ok(core_make_positions(&reqs))
}

#[pyfunction]
fn make_input_mapping(
    table_idxs: Vec<i32>,
    cached_lens: Vec<usize>,
    device_lens: Vec<usize>,
) -> PyResult<Vec<i32>> {
    if table_idxs.len() != cached_lens.len() || cached_lens.len() != device_lens.len() {
        return Err(PyValueError::new_err(
            "table_idxs, cached_lens, and device_lens lengths must match",
        ));
    }
    let mut reqs = make_reqs_for_positions(&cached_lens, &device_lens)?;
    for (req, table_idx) in reqs.iter_mut().zip(table_idxs) {
        req.table_idx = table_idx;
    }
    Ok(core_make_input_mapping(&reqs))
}

#[pyfunction]
fn make_write_mapping(
    table_idxs: Vec<i32>,
    device_lens: Vec<usize>,
    can_decode: Vec<bool>,
) -> PyResult<(Vec<i32>, Vec<i32>)> {
    if table_idxs.len() != device_lens.len() || device_lens.len() != can_decode.len() {
        return Err(PyValueError::new_err(
            "table_idxs, device_lens, and can_decode lengths must match",
        ));
    }
    let mut reqs = Vec::with_capacity(table_idxs.len());
    for idx in 0..table_idxs.len() {
        let decode = can_decode[idx];
        let device_len = device_lens[idx];
        reqs.push(ScheduledReq {
            uid: idx as u64,
            table_idx: table_idxs[idx],
            cached_len: 0,
            device_len,
            max_device_len: device_len + usize::from(decode),
            output_len: usize::from(decode),
            cache_handle: DummyHandle,
            is_chunked: !decode,
        });
    }
    Ok(core_make_write_tuple(&reqs))
}

fn vec_i32_to_bytearray<'py>(py: Python<'py>, values: &[i32]) -> Bound<'py, PyByteArray> {
    let byte_len = std::mem::size_of_val(values);
    let ptr = values.as_ptr().cast::<u8>();
    let bytes = unsafe { std::slice::from_raw_parts(ptr, byte_len) };
    PyByteArray::new(py, bytes)
}

#[pyfunction]
#[allow(clippy::type_complexity)]
fn make_metadata_buffers<'py>(
    py: Python<'py>,
    table_idxs_padded: Vec<i32>,
    cached_lens: Vec<usize>,
    device_lens_padded: Vec<usize>,
    table_idxs: Vec<i32>,
    device_lens: Vec<usize>,
    can_decode: Vec<bool>,
) -> PyResult<(
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
)> {
    if table_idxs_padded.len() != cached_lens.len() || cached_lens.len() != device_lens_padded.len()
    {
        return Err(PyValueError::new_err(
            "table_idxs_padded, cached_lens, and device_lens_padded lengths must match",
        ));
    }

    let mut padded_reqs = make_reqs_for_positions(&cached_lens, &device_lens_padded)?;
    for (req, table_idx) in padded_reqs.iter_mut().zip(table_idxs_padded) {
        req.table_idx = table_idx;
    }
    let positions = core_make_positions(&padded_reqs);
    let input_mapping = core_make_input_mapping(&padded_reqs);

    if table_idxs.len() != device_lens.len() || device_lens.len() != can_decode.len() {
        return Err(PyValueError::new_err(
            "table_idxs, device_lens, and can_decode lengths must match",
        ));
    }
    let mut reqs = Vec::with_capacity(table_idxs.len());
    for idx in 0..table_idxs.len() {
        let decode = can_decode[idx];
        let device_len = device_lens[idx];
        reqs.push(ScheduledReq {
            uid: idx as u64,
            table_idx: table_idxs[idx],
            cached_len: 0,
            device_len,
            max_device_len: device_len + usize::from(decode),
            output_len: usize::from(decode),
            cache_handle: DummyHandle,
            is_chunked: !decode,
        });
    }
    let (write_req_mapping, write_mapping) = core_make_write_tuple(&reqs);

    Ok((
        vec_i32_to_bytearray(py, &positions),
        vec_i32_to_bytearray(py, &input_mapping),
        vec_i32_to_bytearray(py, &write_req_mapping),
        vec_i32_to_bytearray(py, &write_mapping),
    ))
}

#[derive(Clone, Debug)]
struct PrefillHandle;

struct FakePrefillCache {
    available_size: usize,
    cached_len: usize,
    lock_impact: usize,
}

impl PrefillCache for FakePrefillCache {
    type Handle = PrefillHandle;

    fn match_req(
        &mut self,
        _input_ids_without_last: &[i32],
    ) -> Result<CacheMatch<Self::Handle>, String> {
        Ok(CacheMatch {
            handle: PrefillHandle,
            cached_len: self.cached_len,
            match_indices: vec![0; self.cached_len],
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

struct FakePrefillTable {
    available_slots: usize,
    next_idx: i32,
}

impl PrefillTable for FakePrefillTable {
    fn available_size(&self) -> usize {
        self.available_slots
    }

    fn allocate(&mut self) -> Option<i32> {
        if self.available_slots == 0 {
            return None;
        }
        self.available_slots -= 1;
        let out = self.next_idx;
        self.next_idx += 1;
        Some(out)
    }
}

#[pyfunction]
#[pyo3(signature = (
    token_budget,
    reserved_size,
    cache_available_size,
    table_available_size,
    input_len,
    output_len,
    cached_len,
    lock_impact=0
))]
#[allow(clippy::too_many_arguments)]
fn prefill_admission_plan(
    token_budget: usize,
    reserved_size: usize,
    cache_available_size: usize,
    table_available_size: usize,
    input_len: usize,
    output_len: usize,
    cached_len: usize,
    lock_impact: usize,
) -> PyResult<(bool, bool, usize, usize, usize, usize)> {
    let mut cache = FakePrefillCache {
        available_size: cache_available_size,
        cached_len,
        lock_impact,
    };
    let mut table = FakePrefillTable {
        available_slots: table_available_size,
        next_idx: 0,
    };
    let mut adder = PrefillAdder {
        token_budget,
        reserved_size,
        cache: &mut cache,
        table: &mut table,
    };
    let pending = PendingReq {
        uid: 0,
        input_ids: (0..input_len as i32).collect(),
        output_len,
        chunked_req: None,
    };

    let out = adder.try_add_one(&pending).map_err(cache_err)?;
    if let Some(req) = out {
        Ok((
            true,
            req.is_chunked,
            req.cached_len,
            req.device_len,
            adder.token_budget,
            adder.reserved_size,
        ))
    } else {
        Ok((false, false, 0, 0, adder.token_budget, adder.reserved_size))
    }
}

#[pyfunction]
fn ping() -> &'static str {
    "ok"
}

#[pyfunction]
fn core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn mini_sgl_cpu_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(make_positions, m)?)?;
    m.add_function(wrap_pyfunction!(make_input_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(make_write_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(make_metadata_buffers, m)?)?;
    m.add_function(wrap_pyfunction!(prefill_admission_plan, m)?)?;
    m.add_class::<PySamplingParams>()?;
    m.add_class::<PyRadixCacheManager>()?;
    Ok(())
}
