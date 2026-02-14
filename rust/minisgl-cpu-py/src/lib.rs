use minisgl_cpu_core::SamplingParams;
use pyo3::prelude::*;

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
    m.add_class::<PySamplingParams>()?;
    Ok(())
}
