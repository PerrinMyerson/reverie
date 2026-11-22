//! PyO3 bindings for Reverie, exposed to Python as `reverie._reverie` and
//! re-exported by the pure-Python `reverie` package.

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use ariadne::{Config, Label, Report, ReportKind, Source};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyString, PyTuple};
use reverie_cli::embed::{self, EmbedError, ProofEngine};
use reverie_core::CoreError;
use reverie_interp::{IoState, RuntimeError, State, Value};
use reverie_syntax::{Program, SyntaxDiagnostic};

const STRING_PATH: &str = "<string>";

/// A parsed and statically checked Reverie program.
#[pyclass(module = "reverie", name = "Program", frozen)]
struct PyProgram {
    program: Arc<Program>,
    source: Arc<str>,
    path: Option<String>,
    engine: ProofEngine,
}

/// One step of a forward execution timeline.
#[pyclass(module = "reverie", name = "Frame", frozen, get_all)]
struct PyFrame {
    /// Zero-based step index; frame 0 is the entry state.
    step: usize,
    /// Human-readable description of the statement that produced this frame.
    label: String,
    /// Byte range of that statement in the source, if any.
    span: Option<(usize, usize)>,
    /// 1-based line of the statement, if any.
    line: Option<usize>,
    /// 1-based column of the statement, if any.
    column: Option<usize>,
    /// The store after this step, as a plain dict.
    store: Py<PyDict>,
}

#[pymethods]
impl PyFrame {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Frame(step={}, label={:?}, store={})",
            self.step,
            self.label,
            self.store.bind(py).repr()?
        ))
    }
}

#[pymethods]
impl PyProgram {
    /// The program's source text.
    #[getter]
    fn source(&self) -> &str {
        &self.source
    }

    /// The file the program was loaded from, or None for `loads`.
    #[getter]
    fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// The execution engine: "slot" (compiled, default) or "tree".
    #[getter]
    fn engine(&self) -> &'static str {
        engine_name(self.engine)
    }

    /// Run forward from `store` (a mapping, keyword arguments, or both) and
    /// return the final store as a dict.
    #[pyo3(signature = (store = None, /, **vars))]
    fn run(
        &self,
        py: Python<'_>,
        store: Option<&Bound<'_, PyAny>>,
        vars: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        self.execute(py, store, vars, Direction::Forward)
    }

    /// Run backward (the inverse program) from `store` and return the
    /// restored store as a dict.
    #[pyo3(signature = (store = None, /, **vars))]
    fn reverse(
        &self,
        py: Python<'_>,
        store: Option<&Bound<'_, PyAny>>,
        vars: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        self.execute(py, store, vars, Direction::Backward)
    }

    /// Run forward then backward and return the `reverie_roundtrip_proof_v1`
    /// document the CLI writes with `--proof-output`, as a dict.
    #[pyo3(signature = (store = None, /, **vars))]
    fn roundtrip(
        &self,
        py: Python<'_>,
        store: Option<&Bound<'_, PyAny>>,
        vars: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let seeds = seeds_from_py(store, vars)?;
        let program = Arc::clone(&self.program);
        let source = Arc::clone(&self.source);
        let path = self.path.clone().unwrap_or_else(|| STRING_PATH.to_owned());
        let engine = self.engine;
        let proof = py
            .detach(move || {
                embed::roundtrip_proof(Path::new(&path), &source, &program, &seeds, engine)
                    .map(|proof| proof.to_string())
            })
            .map_err(|error| self.embed_error(py, error))?;
        json_loads(py, &proof)
    }

    /// Record every step of a forward run. Returns a list of `Frame`s; the
    /// first is the entry store and the last is the final store.
    #[pyo3(signature = (store = None, /, **vars))]
    fn timeline(
        &self,
        py: Python<'_>,
        store: Option<&Bound<'_, PyAny>>,
        vars: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyList>> {
        let seeds = self.checked_seeds(py, store, vars)?;
        let program = Arc::clone(&self.program);
        let timeline = py
            .detach(move || reverie_interp::build_timeline(&program, state_from_seeds(seeds)))
            .map_err(|error| self.runtime_error(py, &error))?;
        let frames = PyList::empty(py);
        for frame in timeline.frames() {
            let (line, column) = match &frame.span {
                Some(span) => {
                    let (line, column) = line_column(&self.source, span.start);
                    (Some(line), Some(column))
                }
                None => (None, None),
            };
            frames.append(Py::new(
                py,
                PyFrame {
                    step: frame.step,
                    label: frame.label.clone(),
                    span: frame.span.as_ref().map(|span| (span.start, span.end)),
                    line,
                    column,
                    store: store_to_py(py, &frame.state)?.unbind(),
                },
            )?)?;
        }
        Ok(frames.unbind())
    }

    /// Source of the inverse program, in canonical `reverie fmt` style.
    fn inverse_source(&self) -> String {
        reverie_syntax::format_program(&reverie_core::invert_program(&self.program))
    }

    /// Source of this program in canonical `reverie fmt` style.
    fn formatted_source(&self) -> String {
        reverie_syntax::format_program(&self.program)
    }

    fn __repr__(&self) -> String {
        match &self.path {
            Some(path) => format!("<reverie.Program {path:?} engine={:?}>", self.engine()),
            None => format!("<reverie.Program from string engine={:?}>", self.engine()),
        }
    }
}

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Backward,
}

impl PyProgram {
    fn display_path(&self) -> &str {
        self.path.as_deref().unwrap_or(STRING_PATH)
    }

    fn execute(
        &self,
        py: Python<'_>,
        store: Option<&Bound<'_, PyAny>>,
        vars: Option<&Bound<'_, PyDict>>,
        direction: Direction,
    ) -> PyResult<Py<PyDict>> {
        let seeds = self.checked_seeds(py, store, vars)?;
        let program = Arc::clone(&self.program);
        let engine = self.engine;
        let result = py
            .detach(move || {
                let state = state_from_seeds(seeds);
                let io = IoState::new(state, []);
                let options = reverie_interp::ExecutionOptions::default();
                match (engine, direction) {
                    (ProofEngine::Tree, Direction::Forward) => {
                        reverie_interp::execute_io_with_options(&program, io, options)
                    }
                    (ProofEngine::Tree, Direction::Backward) => {
                        reverie_interp::execute_io_backward_with_options(&program, io, options)
                    }
                    (ProofEngine::Slot, Direction::Forward) => {
                        reverie_interp::execute_compiled_io_with_options(&program, io, options)
                    }
                    (ProofEngine::Slot, Direction::Backward) => {
                        reverie_interp::execute_compiled_io_backward_with_options(
                            &program, io, options,
                        )
                    }
                }
            })
            .map_err(|error| self.runtime_error(py, &error))?;
        Ok(store_to_py(py, result.store())?.unbind())
    }

    /// Convert and validate seeds, then re-check the program against the
    /// types inferred from them (as `reverie run --var ...` does).
    fn checked_seeds(
        &self,
        py: Python<'_>,
        store: Option<&Bound<'_, PyAny>>,
        vars: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<(String, Value)>> {
        let seeds = seeds_from_py(store, vars)?;
        let types = embed::seed_types(&self.program, &seeds)
            .map_err(|error| self.embed_error(py, error))?;
        reverie_core::check_program_with_types(&self.program, types)
            .map_err(|error| check_error(py, self.display_path(), &self.source, &error))?;
        Ok(seeds)
    }

    fn runtime_error(&self, py: Python<'_>, error: &RuntimeError) -> PyErr {
        let span = error
            .span()
            .or_else(|| error.labels().first().map(|label| label.span()));
        let mut labels = Vec::new();
        if let Some(span) = error.span() {
            labels.push((span, "runtime failed here".to_owned()));
        }
        labels.extend(
            error
                .labels()
                .iter()
                .map(|label| (label.span(), label.message().to_owned())),
        );
        if labels.is_empty() {
            if let Some(span) = &span {
                labels.push((span.clone(), "runtime failed here".to_owned()));
            }
        }
        reverie_error(
            py,
            "ExecutionError",
            &error.to_string(),
            self.display_path(),
            &self.source,
            span,
            &labels,
        )
    }

    fn embed_error(&self, py: Python<'_>, error: EmbedError) -> PyErr {
        match error {
            EmbedError::Seed(message) => {
                reverie_error(py, "SeedError", &message, "", "", None, &[])
            }
            EmbedError::Check(error) => check_error(py, self.display_path(), &self.source, &error),
            EmbedError::Runtime(error) => self.runtime_error(py, &error),
            EmbedError::Other(message) => {
                reverie_error(py, "ReverieError", &message, "", "", None, &[])
            }
        }
    }
}

fn engine_name(engine: ProofEngine) -> &'static str {
    match engine {
        ProofEngine::Tree => "tree",
        ProofEngine::Slot => "slot",
    }
}

fn parse_engine(engine: &str) -> PyResult<ProofEngine> {
    match engine {
        "slot" => Ok(ProofEngine::Slot),
        "tree" => Ok(ProofEngine::Tree),
        other => Err(PyValueError::new_err(format!(
            "unknown engine {other:?}; expected \"slot\" or \"tree\""
        ))),
    }
}

fn compile(
    py: Python<'_>,
    source: String,
    path: Option<String>,
    engine: ProofEngine,
) -> PyResult<PyProgram> {
    let display = path.clone().unwrap_or_else(|| STRING_PATH.to_owned());
    let program = match reverie_syntax::parse_program(&source) {
        Ok(program) => program,
        Err(diagnostics) => return Err(parse_error(py, &display, &source, &diagnostics)),
    };
    reverie_core::check_program(&program)
        .map_err(|error| check_error(py, &display, &source, &error))?;
    Ok(PyProgram {
        program: Arc::new(program),
        source: Arc::from(source),
        path,
        engine,
    })
}

/// Parse and check the Reverie program in the file at `path`.
#[pyfunction]
#[pyo3(signature = (path, *, engine = "slot"))]
fn load(py: Python<'_>, path: &Bound<'_, PyAny>, engine: &str) -> PyResult<PyProgram> {
    let engine = parse_engine(engine)?;
    let fspath = py.import("os")?.getattr("fspath")?.call1((path,))?;
    let path: String = fspath.extract()?;
    let source = std::fs::read_to_string(&path)
        .map_err(|error| PyOSError::new_err(format!("failed to read `{path}`: {error}")))?;
    compile(py, source, Some(path), engine)
}

/// Parse and check Reverie source text.
#[pyfunction]
#[pyo3(signature = (source, *, engine = "slot"))]
fn loads(py: Python<'_>, source: String, engine: &str) -> PyResult<PyProgram> {
    let engine = parse_engine(engine)?;
    compile(py, source, None, engine)
}

/// Verify a roundtrip proof the way `reverie verify-roundtrip --json` does.
/// `proof` is a proof dict, or a path to a proof JSON file. `source` (a
/// `Program` or source text) replays instead of the proof's recorded file.
#[pyfunction]
#[pyo3(signature = (proof, *, source = None))]
fn verify(
    py: Python<'_>,
    proof: &Bound<'_, PyAny>,
    source: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let source: Option<String> = match source {
        None => None,
        Some(source) if source.is_none() => None,
        Some(source) => match source.cast::<PyProgram>() {
            Ok(program) => Some(program.get().source.to_string()),
            Err(_) => Some(source.extract()?),
        },
    };
    let (text, label) = if proof.is_instance_of::<PyDict>() {
        let json = py.import("json")?;
        let text: String = json.getattr("dumps")?.call1((proof,))?.extract()?;
        (text, "<proof>".to_owned())
    } else {
        let fspath = py.import("os")?.getattr("fspath")?.call1((proof,))?;
        let path: String = fspath.extract()?;
        let text = std::fs::read_to_string(&path)
            .map_err(|error| PyOSError::new_err(format!("failed to read `{path}`: {error}")))?;
        (text, path)
    };
    let proof: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| PyValueError::new_err(format!("proof is not valid JSON: {error}")))?;
    let report = py
        .detach(move || {
            embed::verify_roundtrip_proof(&proof, &label, source.as_deref())
                .map(|report| report.to_string())
        })
        .map_err(|error| {
            reverie_error(py, "ReverieError", &error.to_string(), "", "", None, &[])
        })?;
    json_loads(py, &report)
}

fn json_loads(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("json")?
        .getattr("loads")?
        .call1((text,))?
        .unbind())
}

// ---------------------------------------------------------------------------
// Values

fn seeds_from_py(
    store: Option<&Bound<'_, PyAny>>,
    vars: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<(String, Value)>> {
    let py_store = match store {
        Some(store) if !store.is_none() => {
            let dict = PyDict::new(store.py());
            dict.call_method1("update", (store,))?;
            Some(dict)
        }
        _ => None,
    };
    let mut seeds: Vec<(String, Value)> = Vec::new();
    for source in py_store.as_ref().into_iter().chain(vars) {
        for (key, value) in source.iter() {
            let name: String = key.extract().map_err(|_| {
                seed_error(key.py(), format!("store keys must be strings, got {key}"))
            })?;
            let value = value_from_py(&value, &name)?;
            match seeds.iter_mut().find(|(existing, _)| *existing == name) {
                Some(slot) => slot.1 = value,
                None => seeds.push((name, value)),
            }
        }
    }
    Ok(seeds)
}

fn value_from_py(obj: &Bound<'_, PyAny>, context: &str) -> PyResult<Value> {
    let py = obj.py();
    if obj.is_instance_of::<PyBool>() {
        return Ok(Value::Bool(obj.extract()?));
    }
    let stack_type = py.import("reverie._types")?.getattr("Stack")?;
    if obj.is_instance(&stack_type)? {
        let items = obj.getattr("items")?;
        let mut values = Vec::new();
        for (index, item) in items.try_iter()?.enumerate() {
            let item = item?;
            values.push(int_from_py(&item, &format!("{context}.stack[{index}]"))?);
        }
        return Ok(Value::Stack(values));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let mut values = Vec::new();
        for (index, item) in obj.try_iter()?.enumerate() {
            values.push(value_from_py(&item?, &format!("{context}[{index}]"))?);
        }
        return Ok(Value::Array(values));
    }
    if obj.is_instance_of::<PyString>() {
        return Err(unsupported(obj, context));
    }
    // NumPy arrays and scalars (including `np.bool_`) convert via `tolist()`.
    if obj.hasattr("tolist")? {
        return value_from_py(&obj.call_method0("tolist")?, context);
    }
    if obj.hasattr("__index__")? {
        return Ok(Value::Int(int_from_py(obj, context)?));
    }
    Err(unsupported(obj, context))
}

fn int_from_py(obj: &Bound<'_, PyAny>, context: &str) -> PyResult<i64> {
    if obj.is_instance_of::<PyBool>() || !obj.hasattr("__index__")? {
        return Err(seed_error(
            obj.py(),
            format!(
                "seed `{context}` must be an int, got {}",
                obj.get_type().name()?
            ),
        ));
    }
    let index = obj.call_method0("__index__")?;
    index.extract::<i64>().map_err(|_| {
        seed_error(
            obj.py(),
            format!("seed `{context}` is outside signed 64-bit range"),
        )
    })
}

fn unsupported(obj: &Bound<'_, PyAny>, context: &str) -> PyErr {
    let type_name = obj
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "?".to_owned());
    seed_error(
        obj.py(),
        format!("seed `{context}` must be an int, bool, list, or reverie.Stack, got {type_name}"),
    )
}

fn value_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Int(value) => value.into_pyobject(py)?.into_any(),
        Value::Bool(value) => PyBool::new(py, *value).to_owned().into_any(),
        Value::Array(values) => {
            let list = PyList::empty(py);
            for value in values {
                list.append(value_to_py(py, value)?)?;
            }
            list.into_any()
        }
        Value::Stack(values) => {
            let items = PyList::new(py, values)?;
            py.import("reverie._types")?
                .getattr("Stack")?
                .call1((items,))?
        }
    })
}

fn store_to_py<'py>(py: Python<'py>, state: &State) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, value) in state.store() {
        dict.set_item(name, value_to_py(py, value)?)?;
    }
    Ok(dict)
}

fn state_from_seeds(seeds: Vec<(String, Value)>) -> State {
    State::from_bindings(seeds)
}

// ---------------------------------------------------------------------------
// Errors

fn seed_error(py: Python<'_>, message: String) -> PyErr {
    reverie_error(py, "SeedError", &message, "", "", None, &[])
}

fn parse_error(
    py: Python<'_>,
    path: &str,
    source: &str,
    diagnostics: &[SyntaxDiagnostic],
) -> PyErr {
    let first = diagnostics.first();
    let message = first
        .map(|diagnostic| diagnostic.message.clone())
        .unwrap_or_else(|| "parse error".to_owned());
    let rendered = diagnostics
        .iter()
        .map(|diagnostic| {
            render_report(
                path,
                source,
                &diagnostic.message,
                &[(diagnostic.span.clone(), diagnostic.label.clone())],
            )
        })
        .collect::<String>();
    build_error(
        py,
        "ParseError",
        &message,
        source,
        first.map(|diagnostic| diagnostic.span.clone()),
        &rendered,
    )
}

fn check_error(py: Python<'_>, path: &str, source: &str, error: &CoreError) -> PyErr {
    let labels = if error.labels().is_empty() {
        vec![(error.span(), "semantic check failed here".to_owned())]
    } else {
        error
            .labels()
            .iter()
            .map(|label| (label.span(), label.message().to_owned()))
            .collect()
    };
    reverie_error(
        py,
        "CheckError",
        &error.to_string(),
        path,
        source,
        Some(error.span()),
        &labels,
    )
}

fn reverie_error(
    py: Python<'_>,
    class: &str,
    message: &str,
    path: &str,
    source: &str,
    span: Option<Range<usize>>,
    labels: &[(Range<usize>, String)],
) -> PyErr {
    let rendered = match &span {
        Some(_) if !labels.is_empty() => render_report(path, source, message, labels),
        _ => format!("Error: {message}\n"),
    };
    build_error(py, class, message, source, span, &rendered)
}

fn build_error(
    py: Python<'_>,
    class: &str,
    message: &str,
    source: &str,
    span: Option<Range<usize>>,
    rendered: &str,
) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let cls = py.import("reverie._types")?.getattr(class)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("rendered", rendered)?;
        if let Some(span) = &span {
            let (line, column) = line_column(source, span.start);
            kwargs.set_item("line", line)?;
            kwargs.set_item("column", column)?;
            kwargs.set_item("span", (span.start, span.end))?;
        }
        let instance = cls.call((message,), Some(&kwargs))?;
        Ok(PyErr::from_value(instance))
    };
    build().unwrap_or_else(|error| error)
}

/// Render an ariadne report without ANSI color, as the CLI prints it.
fn render_report(
    path: &str,
    source: &str,
    message: &str,
    labels: &[(Range<usize>, String)],
) -> String {
    let anchor = labels.first().map(|(span, _)| span.clone()).unwrap_or(0..0);
    let mut report = Report::build(ReportKind::Error, (path.to_owned(), anchor))
        .with_config(Config::default().with_color(false))
        .with_message(message);
    for (span, label) in labels {
        report = report.with_label(Label::new((path.to_owned(), span.clone())).with_message(label));
    }
    let mut out = Vec::new();
    if report
        .finish()
        .write((path.to_owned(), Source::from(source)), &mut out)
        .is_err()
    {
        return format!("Error: {message}\n");
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// 1-based line and column (in characters) for a byte offset.
fn line_column(source: &str, offset: usize) -> (usize, usize) {
    let mut offset = offset.min(source.len());
    while !source.is_char_boundary(offset) {
        offset -= 1;
    }
    let before = &source[..offset];
    let line = before.matches('\n').count() + 1;
    let line_start = before.rfind('\n').map_or(0, |index| index + 1);
    let column = before[line_start..].chars().count() + 1;
    (line, column)
}

#[pymodule]
fn _reverie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProgram>()?;
    m.add_class::<PyFrame>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(loads, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
