use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::ops::Range;

use reverie_syntax::{
    BinaryOp, Expr, GlobalDecl, Place, Proc, Program, Spanned, SpannedExpr, SpannedStmt, Stmt,
    TypeExpr, UnaryOp, UpdateOp,
};
use thiserror::Error;

mod compiled;

pub use compiled::{
    CompiledProgram, execute_compiled, execute_compiled_backward, execute_compiled_io,
    execute_compiled_io_backward,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Array(Vec<Value>),
    Stack(Vec<i64>),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(value) => write!(f, "{value}"),
            Self::Bool(value) => write!(f, "{value}"),
            Self::Array(values) => {
                write!(f, "[")?;
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{value}")?;
                }
                write!(f, "]")
            }
            Self::Stack(values) => {
                if values.is_empty() {
                    write!(f, "nil")
                } else {
                    write!(f, "stack[")?;
                    for (index, value) in values.iter().enumerate() {
                        if index > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{value}")?;
                    }
                    write!(f, "]")
                }
            }
        }
    }
}

pub(crate) fn format_show_observation(name: &str, value: &Value) -> String {
    format!("{name} = {}", format_show_value(value))
}

pub(crate) fn format_printf_observation(
    format: &str,
    args: &[i64],
    span: Range<usize>,
) -> Result<String, RuntimeError> {
    let mut rendered = String::new();
    let mut args = args.iter();
    let mut chars = format.chars();

    while let Some(ch) = chars.next() {
        if ch != '%' {
            rendered.push(ch);
            continue;
        }

        let Some(specifier) = chars.next() else {
            return Err(RuntimeError::at("incomplete printf format specifier", span));
        };

        match specifier {
            '%' => rendered.push('%'),
            'd' => {
                let value = args.next().ok_or_else(|| {
                    RuntimeError::at("printf expected another argument", span.clone())
                })?;
                rendered.push_str(&value.to_string());
            }
            other => {
                return Err(RuntimeError::at(
                    format!("unsupported printf format specifier `%{other}`"),
                    span,
                ));
            }
        }
    }

    if args.next().is_some() {
        return Err(RuntimeError::at("printf received too many arguments", span));
    }

    Ok(rendered)
}

fn format_show_value(value: &Value) -> String {
    match value {
        Value::Stack(values) => {
            let inner = values
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("<{inner}]")
        }
        other => other.to_string(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct State {
    store: BTreeMap<String, Value>,
}

impl State {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn from_bindings(bindings: impl IntoIterator<Item = (String, Value)>) -> Self {
        Self {
            store: bindings.into_iter().collect(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.store.get(name)
    }

    pub fn insert(&mut self, name: impl Into<String>, value: Value) {
        self.store.insert(name.into(), value);
    }

    pub fn remove(&mut self, name: &str) -> Option<Value> {
        self.store.remove(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.store.contains_key(name)
    }

    pub fn store(&self) -> &BTreeMap<String, Value> {
        &self.store
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;

        for (index, (name, value)) in self.store.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }

            write!(f, "{name} = {value}")?;
        }

        write!(f, "}}")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoState {
    store: State,
    input: VecDeque<Value>,
    output: Vec<Value>,
    observations: Vec<String>,
}

impl IoState {
    pub fn new(store: State, input: impl IntoIterator<Item = Value>) -> Self {
        Self {
            store,
            input: input.into_iter().collect(),
            output: Vec::new(),
            observations: Vec::new(),
        }
    }

    pub fn with_output(
        store: State,
        input: impl IntoIterator<Item = Value>,
        output: impl IntoIterator<Item = Value>,
    ) -> Self {
        Self {
            store,
            input: input.into_iter().collect(),
            output: output.into_iter().collect(),
            observations: Vec::new(),
        }
    }

    pub fn store(&self) -> &State {
        &self.store
    }

    pub fn input(&self) -> &VecDeque<Value> {
        &self.input
    }

    pub fn output(&self) -> &[Value] {
        &self.output
    }

    pub fn observations(&self) -> &[String] {
        &self.observations
    }

    pub fn into_store(self) -> State {
        self.store
    }

    pub(crate) fn into_parts(self) -> (State, VecDeque<Value>, Vec<Value>, Vec<String>) {
        (self.store, self.input, self.output, self.observations)
    }

    pub(crate) fn from_parts(
        store: State,
        input: VecDeque<Value>,
        output: Vec<Value>,
        observations: Vec<String>,
    ) -> Self {
        Self {
            store,
            input,
            output,
            observations,
        }
    }
}

fn initialize_globals(globals: &[GlobalDecl], state: &mut State) -> Result<(), RuntimeError> {
    for global in globals {
        if let Some(value) = state.get(&global.name) {
            validate_declared_value(
                &global.name,
                &global.dims,
                global.ty.as_ref().map(|ty| &ty.node),
                value,
            )?;
            continue;
        }

        let value = initial_global_value(global, state)?;
        state.insert(global.name.clone(), value);
    }

    Ok(())
}

fn initial_global_value(global: &GlobalDecl, state: &State) -> Result<Value, RuntimeError> {
    let value = if let Some(init) = &global.init {
        eval(init, state)?
    } else {
        default_value(&global.dims, global.ty.as_ref().map(|ty| &ty.node))?
    };
    validate_declared_value(
        &global.name,
        &global.dims,
        global.ty.as_ref().map(|ty| &ty.node),
        &value,
    )?;
    Ok(value)
}

fn default_value(dims: &[usize], ty: Option<&TypeExpr>) -> Result<Value, RuntimeError> {
    if dims.contains(&0) {
        return Err(RuntimeError::new("declaration length must be at least 1"));
    }

    if let Some((len, rest)) = dims.split_first() {
        let element_ty = match ty {
            Some(TypeExpr::Array { element }) => Some(&element.node),
            other => other,
        };
        return (0..*len)
            .map(|_| default_value(rest, element_ty))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array);
    }

    match ty {
        Some(TypeExpr::Stack) => Ok(Value::Stack(Vec::new())),
        Some(TypeExpr::Array { element }) => default_value(&[1], Some(&element.node)),
        Some(TypeExpr::Bool) => Ok(Value::Bool(false)),
        Some(TypeExpr::Int { .. }) | None => Ok(Value::Int(0)),
    }
}

fn validate_declared_shape(name: &str, dims: &[usize], value: &Value) -> Result<(), RuntimeError> {
    let Some((len, rest)) = dims.split_first() else {
        return Ok(());
    };

    match value {
        Value::Array(values) if values.len() == *len => {
            for element in values {
                validate_declared_shape(name, rest, element)?;
            }
            Ok(())
        }
        Value::Array(values) => Err(RuntimeError::new(format!(
            "declaration `{name}` expected array length {len}, found {}",
            values.len()
        ))),
        other => Err(RuntimeError::new(format!(
            "declaration `{name}` expected array length {len}, found {other}"
        ))),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("runtime error: {message}")]
pub struct RuntimeError {
    message: String,
    span: Option<Range<usize>>,
    labels: Vec<RuntimeLabel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeLabel {
    span: Range<usize>,
    message: String,
}

impl RuntimeLabel {
    pub fn new(span: Range<usize>, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
        }
    }

    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl RuntimeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
            labels: Vec::new(),
        }
    }

    pub fn at(message: impl Into<String>, span: Range<usize>) -> Self {
        Self {
            message: message.into(),
            span: Some(span),
            labels: Vec::new(),
        }
    }

    pub fn span(&self) -> Option<Range<usize>> {
        self.span.clone()
    }

    pub fn labels(&self) -> &[RuntimeLabel] {
        &self.labels
    }

    pub(crate) fn with_span_if_missing(mut self, span: Range<usize>) -> Self {
        if self.span.is_none() {
            self.span = Some(span);
        }
        self
    }

    pub(crate) fn with_context_label(
        mut self,
        span: Range<usize>,
        message: impl Into<String>,
    ) -> Self {
        self.labels.push(RuntimeLabel::new(span, message));
        self
    }

    fn with_call_context(
        mut self,
        name: &str,
        direction: Direction,
        call_span: Range<usize>,
    ) -> Self {
        self.labels.push(RuntimeLabel::new(
            call_span,
            format!("while {} `{name}` here", direction.gerund()),
        ));
        self
    }
}

pub fn run_program(program: &Program) -> Result<State, RuntimeError> {
    execute(program, State::empty())
}

pub fn execute(program: &Program, state: State) -> Result<State, RuntimeError> {
    execute_io(program, IoState::new(state, [])).map(IoState::into_store)
}

pub fn execute_io(program: &Program, mut state: IoState) -> Result<IoState, RuntimeError> {
    let runtime = Runtime::new(program)?;
    initialize_globals(&program.globals, &mut state.store)?;
    let mut tapes = Tapes {
        input: &mut state.input,
        output: &mut state.output,
        observations: &mut state.observations,
    };
    state.store = runtime.execute_stmt(&program.body, state.store, &mut tapes)?;
    Ok(state)
}

pub fn execute_backward(program: &Program, state: State) -> Result<State, RuntimeError> {
    execute_io_backward(program, IoState::new(state, [])).map(IoState::into_store)
}

pub fn execute_io_backward(program: &Program, state: IoState) -> Result<IoState, RuntimeError> {
    let inverse = reverie_core::invert_program(program);
    execute_io(&inverse, state)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timeline {
    frames: Vec<TimelineFrame>,
}

impl Timeline {
    pub fn frames(&self) -> &[TimelineFrame] {
        &self.frames
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn final_state(&self) -> Option<&State> {
        self.frames.last().map(|frame| &frame.state)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineFrame {
    pub step: usize,
    pub label: String,
    pub span: Option<Range<usize>>,
    pub state: State,
}

pub fn build_timeline(program: &Program, state: State) -> Result<Timeline, RuntimeError> {
    build_timeline_io(program, IoState::new(state, []))
}

pub fn build_timeline_io(program: &Program, mut state: IoState) -> Result<Timeline, RuntimeError> {
    let runtime = Runtime::new(program)?;
    initialize_globals(&program.globals, &mut state.store)?;
    let mut builder = TimelineBuilder::new(state.store.clone());
    let mut tapes = Tapes {
        input: &mut state.input,
        output: &mut state.output,
        observations: &mut state.observations,
    };
    runtime.trace_stmt(&program.body, state.store, &mut tapes, &mut builder)?;
    Ok(builder.finish())
}

struct TimelineBuilder {
    frames: Vec<TimelineFrame>,
}

impl TimelineBuilder {
    fn new(initial: State) -> Self {
        Self {
            frames: vec![TimelineFrame {
                step: 0,
                label: "start".to_owned(),
                span: None,
                state: initial,
            }],
        }
    }

    fn push(&mut self, label: impl Into<String>, span: Option<Range<usize>>, state: &State) {
        self.frames.push(TimelineFrame {
            step: self.frames.len(),
            label: label.into(),
            span,
            state: state.clone(),
        });
    }

    fn finish(self) -> Timeline {
        Timeline {
            frames: self.frames,
        }
    }
}

struct Runtime<'a> {
    globals: &'a [GlobalDecl],
    procedures: BTreeMap<&'a str, &'a Proc>,
}

struct Tapes<'a> {
    input: &'a mut VecDeque<Value>,
    output: &'a mut Vec<Value>,
    observations: &'a mut Vec<String>,
}

impl<'a> Runtime<'a> {
    fn new(program: &'a Program) -> Result<Self, RuntimeError> {
        let mut procedures = BTreeMap::new();
        for procedure in &program.procedures {
            if procedures
                .insert(procedure.name.as_str(), procedure)
                .is_some()
            {
                return Err(RuntimeError::new(format!(
                    "duplicate procedure `{}`",
                    procedure.name
                )));
            }
        }

        Ok(Self {
            globals: &program.globals,
            procedures,
        })
    }

    fn execute_stmt(
        &self,
        statement: &SpannedStmt,
        state: State,
        tapes: &mut Tapes<'_>,
    ) -> Result<State, RuntimeError> {
        self.execute_stmt_inner(statement, state, tapes)
            .map_err(|error| error.with_span_if_missing(statement.span.clone()))
    }

    fn execute_stmt_inner(
        &self,
        statement: &SpannedStmt,
        mut state: State,
        tapes: &mut Tapes<'_>,
    ) -> Result<State, RuntimeError> {
        match &statement.node {
            Stmt::Skip => Ok(state),
            Stmt::Assert { condition } => {
                assert_condition(condition, &state, true, "assertion")?;
                Ok(state)
            }
            Stmt::Seq(statements) => {
                for statement in statements {
                    state = self.execute_stmt(statement, state, tapes)?;
                }

                Ok(state)
            }
            Stmt::Update { target, op, expr } => {
                ensure_update_rhs_does_not_read_target(target, expr, &state)?;
                let next = updated_value(target, *op, expr, &state)?;
                assign_place(target, &mut state, next)?;
                Ok(state)
            }
            Stmt::Swap { left, right } => {
                let left_value = eval_place(left, &state)?;
                let right_value = eval_place(right, &state)?;

                assign_place(left, &mut state, right_value)?;
                assign_place(right, &mut state, left_value)?;
                Ok(state)
            }
            Stmt::Push { source, stack } => {
                push_stack(source, stack, &mut state)?;
                Ok(state)
            }
            Stmt::Pop { target, stack } => {
                pop_stack(target, stack, &mut state)?;
                Ok(state)
            }
            Stmt::If {
                entry,
                then_branch,
                else_branch,
                exit,
            } => {
                let entry_value = eval_bool(entry, &state)?;
                if entry_value {
                    state = self.execute_stmt(then_branch, state, tapes)?;
                    assert_condition(exit, &state, true, "if exit assertion")?;
                } else {
                    state = self.execute_stmt(else_branch, state, tapes)?;
                    assert_condition(exit, &state, false, "if exit assertion")?;
                }

                Ok(state)
            }
            Stmt::Loop {
                entry,
                body,
                step,
                exit,
            } => {
                assert_condition(entry, &state, true, "loop entry assertion")?;

                loop {
                    state = self.execute_stmt(body, state, tapes)?;
                    if eval_bool(exit, &state)? {
                        return Ok(state);
                    }

                    state = self.execute_stmt(step, state, tapes)?;
                    assert_condition(entry, &state, false, "loop re-entry assertion")?;
                }
            }
            Stmt::Iterate {
                name,
                start,
                step,
                end,
                body,
            } => self.execute_iterate(
                IterateRuntime {
                    name,
                    start,
                    step,
                    end,
                    body,
                },
                statement.span.clone(),
                state,
                tapes,
            ),
            Stmt::Call { name, args } => self.execute_procedure(
                name,
                args,
                statement.span.clone(),
                state,
                Direction::Call,
                tapes,
            ),
            Stmt::Uncall { name, args } => self.execute_procedure(
                name,
                args,
                statement.span.clone(),
                state,
                Direction::Uncall,
                tapes,
            ),
            Stmt::Read { target } => {
                let next = tapes.input.pop_front().ok_or_else(|| {
                    RuntimeError::at("read expected an input tape value", statement.span.clone())
                })?;
                let previous = eval_place(target, &state)?;
                assign_place(target, &mut state, next)?;
                tapes.output.push(previous);
                Ok(state)
            }
            Stmt::Unread { target } => {
                let previous = tapes.output.pop().ok_or_else(|| {
                    RuntimeError::at(
                        "unread expected an output tape value",
                        statement.span.clone(),
                    )
                })?;
                let current = eval_place(target, &state)?;
                assign_place(target, &mut state, previous)?;
                tapes.input.push_front(current);
                Ok(state)
            }
            Stmt::Write { source } => {
                tapes.output.push(eval_place(source, &state)?);
                Ok(state)
            }
            Stmt::Unwrite { source } => {
                let expected = eval_place(source, &state)?;
                let actual = tapes.output.pop().ok_or_else(|| {
                    RuntimeError::at(
                        "unwrite expected an output tape value",
                        statement.span.clone(),
                    )
                })?;
                if actual == expected {
                    Ok(state)
                } else {
                    Err(RuntimeError::at(
                        format!("unwrite expected {expected}, found {actual}"),
                        statement.span.clone(),
                    ))
                }
            }
            Stmt::Show { targets } => {
                for target in targets {
                    let value = state.get(target).cloned().ok_or_else(|| {
                        RuntimeError::at(
                            format!("undefined variable `{target}`"),
                            statement.span.clone(),
                        )
                    })?;
                    tapes
                        .observations
                        .push(format_show_observation(target, &value));
                }
                Ok(state)
            }
            Stmt::Printf { format, args } => {
                let values = args
                    .iter()
                    .map(|arg| eval_int(arg, &state))
                    .collect::<Result<Vec<_>, _>>()?;
                tapes.observations.push(format_printf_observation(
                    format,
                    &values,
                    statement.span.clone(),
                )?);
                Ok(state)
            }
            Stmt::Declare {
                name,
                ty,
                dims,
                init,
                ..
            } => {
                declare_value(name, ty.node.clone(), dims, init.as_ref(), &mut state)?;
                Ok(state)
            }
            Stmt::Local {
                name,
                ty,
                dims,
                refinement,
                init,
                body,
                delocal_name,
                delocal_ty,
                delocal_dims,
                delocal_refinement,
                delocal,
            } => self.execute_local(
                LocalRuntime {
                    name,
                    ty: ty.as_ref().map(|ty| &ty.node),
                    dims,
                    refinement: refinement.as_ref(),
                    init,
                    body,
                    delocal_name,
                    delocal_ty: delocal_ty.as_ref().map(|ty| &ty.node),
                    delocal_dims,
                    delocal_refinement: delocal_refinement.as_ref(),
                    delocal,
                },
                state,
                tapes,
            ),
        }
    }

    fn trace_stmt(
        &self,
        statement: &SpannedStmt,
        state: State,
        tapes: &mut Tapes<'_>,
        timeline: &mut TimelineBuilder,
    ) -> Result<State, RuntimeError> {
        self.trace_stmt_inner(statement, state, tapes, timeline)
            .map_err(|error| error.with_span_if_missing(statement.span.clone()))
    }

    fn trace_stmt_inner(
        &self,
        statement: &SpannedStmt,
        mut state: State,
        tapes: &mut Tapes<'_>,
        timeline: &mut TimelineBuilder,
    ) -> Result<State, RuntimeError> {
        match &statement.node {
            Stmt::Skip => {
                timeline.push("skip", Some(statement.span.clone()), &state);
                Ok(state)
            }
            Stmt::Assert { condition } => {
                assert_condition(condition, &state, true, "assertion")?;
                timeline.push("assert", Some(statement.span.clone()), &state);
                Ok(state)
            }
            Stmt::Seq(statements) => {
                for statement in statements {
                    state = self.trace_stmt(statement, state, tapes, timeline)?;
                }

                Ok(state)
            }
            Stmt::Update { target, op, expr } => {
                ensure_update_rhs_does_not_read_target(target, expr, &state)?;
                let next = updated_value(target, *op, expr, &state)?;
                assign_place(target, &mut state, next)?;
                timeline.push(
                    format!("{} {}", display_place(target), update_label(*op)),
                    Some(statement.span.clone()),
                    &state,
                );
                Ok(state)
            }
            Stmt::Swap { left, right } => {
                let left_value = eval_place(left, &state)?;
                let right_value = eval_place(right, &state)?;

                assign_place(left, &mut state, right_value)?;
                assign_place(right, &mut state, left_value)?;
                timeline.push(
                    format!("{} <=> {}", display_place(left), display_place(right)),
                    Some(statement.span.clone()),
                    &state,
                );
                Ok(state)
            }
            Stmt::Push { source, stack } => {
                push_stack(source, stack, &mut state)?;
                timeline.push(
                    format!("push({}, {stack})", display_place(source)),
                    Some(statement.span.clone()),
                    &state,
                );
                Ok(state)
            }
            Stmt::Pop { target, stack } => {
                pop_stack(target, stack, &mut state)?;
                timeline.push(
                    format!("pop({}, {stack})", display_place(target)),
                    Some(statement.span.clone()),
                    &state,
                );
                Ok(state)
            }
            Stmt::If {
                entry,
                then_branch,
                else_branch,
                exit,
            } => {
                let entry_value = eval_bool(entry, &state)?;
                timeline.push(
                    if entry_value { "if then" } else { "if else" },
                    Some(statement.span.clone()),
                    &state,
                );
                if entry_value {
                    state = self.trace_stmt(then_branch, state, tapes, timeline)?;
                    assert_condition(exit, &state, true, "if exit assertion")?;
                } else {
                    state = self.trace_stmt(else_branch, state, tapes, timeline)?;
                    assert_condition(exit, &state, false, "if exit assertion")?;
                }

                Ok(state)
            }
            Stmt::Loop {
                entry,
                body,
                step,
                exit,
            } => {
                assert_condition(entry, &state, true, "loop entry assertion")?;
                timeline.push("loop enter", Some(statement.span.clone()), &state);

                loop {
                    state = self.trace_stmt(body, state, tapes, timeline)?;
                    if eval_bool(exit, &state)? {
                        timeline.push("loop exit", Some(statement.span.clone()), &state);
                        return Ok(state);
                    }

                    state = self.trace_stmt(step, state, tapes, timeline)?;
                    assert_condition(entry, &state, false, "loop re-entry assertion")?;
                    timeline.push("loop repeat", Some(statement.span.clone()), &state);
                }
            }
            Stmt::Iterate {
                name,
                start,
                step,
                end,
                body,
            } => self.trace_iterate(
                IterateRuntime {
                    name,
                    start,
                    step,
                    end,
                    body,
                },
                statement.span.clone(),
                state,
                tapes,
                timeline,
            ),
            Stmt::Call { name, args } => self.trace_procedure(
                ProcedureTrace {
                    name,
                    args,
                    call_span: statement.span.clone(),
                    direction: Direction::Call,
                },
                state,
                tapes,
                timeline,
            ),
            Stmt::Uncall { name, args } => self.trace_procedure(
                ProcedureTrace {
                    name,
                    args,
                    call_span: statement.span.clone(),
                    direction: Direction::Uncall,
                },
                state,
                tapes,
                timeline,
            ),
            Stmt::Read { target } => {
                let next = tapes.input.pop_front().ok_or_else(|| {
                    RuntimeError::at("read expected an input tape value", statement.span.clone())
                })?;
                let previous = eval_place(target, &state)?;
                assign_place(target, &mut state, next.clone())?;
                tapes.output.push(previous);
                timeline.push(
                    format!("read {} <- {next}", display_place(target)),
                    Some(statement.span.clone()),
                    &state,
                );
                Ok(state)
            }
            Stmt::Unread { target } => {
                let previous = tapes.output.pop().ok_or_else(|| {
                    RuntimeError::at(
                        "unread expected an output tape value",
                        statement.span.clone(),
                    )
                })?;
                let current = eval_place(target, &state)?;
                assign_place(target, &mut state, previous)?;
                tapes.input.push_front(current.clone());
                timeline.push(
                    format!("unread {} -> {current}", display_place(target)),
                    Some(statement.span.clone()),
                    &state,
                );
                Ok(state)
            }
            Stmt::Write { source } => {
                let value = eval_place(source, &state)?;
                tapes.output.push(value.clone());
                timeline.push(
                    format!("write {} = {value}", display_place(source)),
                    Some(statement.span.clone()),
                    &state,
                );
                Ok(state)
            }
            Stmt::Unwrite { source } => {
                let expected = eval_place(source, &state)?;
                let actual = tapes.output.pop().ok_or_else(|| {
                    RuntimeError::at(
                        "unwrite expected an output tape value",
                        statement.span.clone(),
                    )
                })?;
                if actual != expected {
                    return Err(RuntimeError::at(
                        format!("unwrite expected {expected}, found {actual}"),
                        statement.span.clone(),
                    ));
                }
                timeline.push(
                    format!("unwrite {} = {expected}", display_place(source)),
                    Some(statement.span.clone()),
                    &state,
                );
                Ok(state)
            }
            Stmt::Show { targets } => {
                for target in targets {
                    let value = state.get(target).cloned().ok_or_else(|| {
                        RuntimeError::at(
                            format!("undefined variable `{target}`"),
                            statement.span.clone(),
                        )
                    })?;
                    timeline.push(
                        format!("show {}", format_show_observation(target, &value)),
                        Some(statement.span.clone()),
                        &state,
                    );
                }
                Ok(state)
            }
            Stmt::Printf { format, args } => {
                let values = args
                    .iter()
                    .map(|arg| eval_int(arg, &state))
                    .collect::<Result<Vec<_>, _>>()?;
                let observation =
                    format_printf_observation(format, &values, statement.span.clone())?;
                timeline.push(
                    format!("printf {observation:?}"),
                    Some(statement.span.clone()),
                    &state,
                );
                Ok(state)
            }
            Stmt::Declare {
                name,
                ty,
                dims,
                init,
                ..
            } => {
                declare_value(name, ty.node.clone(), dims, init.as_ref(), &mut state)?;
                timeline.push(
                    format!("declare {name}"),
                    Some(statement.span.clone()),
                    &state,
                );
                Ok(state)
            }
            Stmt::Local {
                name,
                ty,
                dims,
                refinement,
                init,
                body,
                delocal_name,
                delocal_ty,
                delocal_dims,
                delocal_refinement,
                delocal,
            } => self.trace_local(
                LocalRuntime {
                    name,
                    ty: ty.as_ref().map(|ty| &ty.node),
                    dims,
                    refinement: refinement.as_ref(),
                    init,
                    body,
                    delocal_name,
                    delocal_ty: delocal_ty.as_ref().map(|ty| &ty.node),
                    delocal_dims,
                    delocal_refinement: delocal_refinement.as_ref(),
                    delocal,
                },
                statement.span.clone(),
                state,
                tapes,
                timeline,
            ),
        }
    }

    fn execute_procedure(
        &self,
        name: &str,
        args: &[Place],
        call_span: Range<usize>,
        mut caller_state: State,
        direction: Direction,
        tapes: &mut Tapes<'_>,
    ) -> Result<State, RuntimeError> {
        let procedure = self
            .procedures
            .get(name)
            .ok_or_else(|| RuntimeError::new(format!("unknown procedure `{name}`")))?;

        if procedure.params.len() != args.len() {
            return Err(RuntimeError::at(
                format!(
                    "procedure `{name}` expects {} argument(s), found {}",
                    procedure.params.len(),
                    args.len()
                ),
                procedure.span.clone(),
            ));
        }

        let frozen_args = freeze_call_args(args, &caller_state)?;
        let args = frozen_args.as_slice();
        ensure_unique_args(name, args)?;

        let mut local_state = State::empty();
        let mut copied_globals = Vec::new();
        let result: Result<State, RuntimeError> = (|| {
            for global in self.globals {
                if procedure
                    .params
                    .iter()
                    .any(|param| param.name == global.name)
                    || args.iter().any(|arg| arg.name == global.name)
                {
                    continue;
                }

                if let Some(value) = caller_state.get(&global.name).cloned() {
                    copied_globals.push(global.name.clone());
                    local_state.insert(global.name.clone(), value);
                }
            }

            for (param, arg) in procedure.params.iter().zip(args) {
                let value = eval_place(arg, &caller_state)?;
                local_state.insert(param.name.clone(), value);
            }
            for param in &procedure.params {
                assert_refinement(&param.name, param.refinement.as_ref(), &local_state)?;
            }

            local_state = match direction {
                Direction::Call => self.execute_stmt(&procedure.body, local_state, tapes)?,
                Direction::Uncall => {
                    let inverse = reverie_core::invert_stmt(&procedure.body);
                    self.execute_stmt(&inverse, local_state, tapes)?
                }
            };
            for param in &procedure.params {
                assert_refinement(&param.name, param.refinement.as_ref(), &local_state)?;
            }

            Ok(local_state)
        })();

        local_state =
            result.map_err(|error| error.with_call_context(name, direction, call_span.clone()))?;

        for (param, arg) in procedure.params.iter().zip(args) {
            let value = local_state.get(&param.name).cloned().ok_or_else(|| {
                RuntimeError::new(format!(
                    "procedure `{name}` lost parameter `{}`",
                    param.name
                ))
            })?;
            assign_place(arg, &mut caller_state, value)?;
        }

        for global in self.globals {
            if !copied_globals.contains(&global.name) {
                continue;
            }

            let value = local_state.get(&global.name).cloned().ok_or_else(|| {
                RuntimeError::new(format!("procedure `{name}` lost global `{}`", global.name))
            })?;
            caller_state.insert(global.name.clone(), value);
        }

        Ok(caller_state)
    }

    fn trace_procedure(
        &self,
        procedure_call: ProcedureTrace<'_>,
        mut caller_state: State,
        tapes: &mut Tapes<'_>,
        timeline: &mut TimelineBuilder,
    ) -> Result<State, RuntimeError> {
        let ProcedureTrace {
            name,
            args,
            call_span,
            direction,
        } = procedure_call;
        let procedure = self
            .procedures
            .get(name)
            .ok_or_else(|| RuntimeError::new(format!("unknown procedure `{name}`")))?;

        if procedure.params.len() != args.len() {
            return Err(RuntimeError::at(
                format!(
                    "procedure `{name}` expects {} argument(s), found {}",
                    procedure.params.len(),
                    args.len()
                ),
                procedure.span.clone(),
            ));
        }

        let frozen_args = freeze_call_args(args, &caller_state)?;
        let args = frozen_args.as_slice();
        ensure_unique_args(name, args)?;

        let mut local_state = State::empty();
        let mut copied_globals = Vec::new();
        let result: Result<State, RuntimeError> = (|| {
            for global in self.globals {
                if procedure
                    .params
                    .iter()
                    .any(|param| param.name == global.name)
                    || args.iter().any(|arg| arg.name == global.name)
                {
                    continue;
                }

                if let Some(value) = caller_state.get(&global.name).cloned() {
                    copied_globals.push(global.name.clone());
                    local_state.insert(global.name.clone(), value);
                }
            }

            for (param, arg) in procedure.params.iter().zip(args) {
                let value = eval_place(arg, &caller_state)?;
                local_state.insert(param.name.clone(), value);
            }
            for param in &procedure.params {
                assert_refinement(&param.name, param.refinement.as_ref(), &local_state)?;
            }

            timeline.push(
                format!("{} {name} enter", direction.verb()),
                Some(call_span.clone()),
                &local_state,
            );

            local_state = match direction {
                Direction::Call => {
                    self.trace_stmt(&procedure.body, local_state, tapes, timeline)?
                }
                Direction::Uncall => {
                    let inverse = reverie_core::invert_stmt(&procedure.body);
                    self.trace_stmt(&inverse, local_state, tapes, timeline)?
                }
            };
            for param in &procedure.params {
                assert_refinement(&param.name, param.refinement.as_ref(), &local_state)?;
            }

            Ok(local_state)
        })();

        local_state =
            result.map_err(|error| error.with_call_context(name, direction, call_span.clone()))?;

        for (param, arg) in procedure.params.iter().zip(args) {
            let value = local_state.get(&param.name).cloned().ok_or_else(|| {
                RuntimeError::new(format!(
                    "procedure `{name}` lost parameter `{}`",
                    param.name
                ))
            })?;
            assign_place(arg, &mut caller_state, value)?;
        }

        for global in self.globals {
            if !copied_globals.contains(&global.name) {
                continue;
            }

            let value = local_state.get(&global.name).cloned().ok_or_else(|| {
                RuntimeError::new(format!("procedure `{name}` lost global `{}`", global.name))
            })?;
            caller_state.insert(global.name.clone(), value);
        }

        timeline.push(
            format!("{} {name} exit", direction.verb()),
            Some(call_span),
            &caller_state,
        );

        Ok(caller_state)
    }

    fn execute_local(
        &self,
        local: LocalRuntime<'_>,
        mut state: State,
        tapes: &mut Tapes<'_>,
    ) -> Result<State, RuntimeError> {
        if local.name != local.delocal_name {
            return Err(RuntimeError::new(format!(
                "local `{}` must be destroyed by matching `delocal {}`",
                local.name, local.name
            )));
        }

        if state.contains(local.name) {
            return Err(RuntimeError::new(format!(
                "local `{}` would shadow an existing variable",
                local.name
            )));
        }

        let init_value = eval(local.init, &state)?;
        validate_local_value(local.name, local.dims, local.ty, &init_value)?;
        state.insert(local.name.to_owned(), init_value);
        assert_refinement(local.name, local.refinement, &state)?;
        state = self.execute_stmt(local.body, state, tapes)?;
        assert_refinement(local.name, local.refinement, &state)?;

        let actual = state
            .get(local.name)
            .cloned()
            .ok_or_else(|| RuntimeError::new(format!("local `{}` disappeared", local.name)))?;
        let expected = eval(local.delocal, &state)?;
        validate_local_value(local.name, local.dims, local.ty, &actual)?;
        validate_local_value(local.name, local.dims, local.ty, &expected)?;
        validate_local_value(local.name, local.delocal_dims, local.delocal_ty, &actual)?;
        validate_local_value(local.name, local.delocal_dims, local.delocal_ty, &expected)?;
        if actual != expected {
            return Err(RuntimeError::at(
                format!(
                    "delocal `{}` expected {expected}, found {actual}",
                    local.name
                ),
                local.delocal.span.clone(),
            ));
        }
        assert_refinement(local.name, local.delocal_refinement, &state)?;

        state.remove(local.name);
        Ok(state)
    }

    fn execute_iterate(
        &self,
        iterate: IterateRuntime<'_>,
        span: Range<usize>,
        mut state: State,
        tapes: &mut Tapes<'_>,
    ) -> Result<State, RuntimeError> {
        let bounds = eval_iterate_bounds(iterate, &state, span.clone())?;
        if state.contains(iterate.name) {
            return Err(RuntimeError::new(format!(
                "iterate variable `{}` would shadow an existing variable",
                iterate.name
            )));
        }

        state.insert(iterate.name.to_owned(), Value::Int(bounds.start));
        let mut current = bounds.start;
        while iterate_contains(current, bounds.end, bounds.step) {
            state.insert(iterate.name.to_owned(), Value::Int(current));
            state = self.execute_stmt(iterate.body, state, tapes)?;
            assert_iterate_variable(iterate.name, current, &state, span.clone())?;
            let Some(next) = next_iterate_value(current, bounds.end, bounds.step, span.clone())?
            else {
                break;
            };
            current = next;
        }
        state.remove(iterate.name);
        Ok(state)
    }

    fn trace_local(
        &self,
        local: LocalRuntime<'_>,
        span: Range<usize>,
        mut state: State,
        tapes: &mut Tapes<'_>,
        timeline: &mut TimelineBuilder,
    ) -> Result<State, RuntimeError> {
        if local.name != local.delocal_name {
            return Err(RuntimeError::new(format!(
                "local `{}` must be destroyed by matching `delocal {}`",
                local.name, local.name
            )));
        }

        if state.contains(local.name) {
            return Err(RuntimeError::new(format!(
                "local `{}` would shadow an existing variable",
                local.name
            )));
        }

        let init_value = eval(local.init, &state)?;
        validate_local_value(local.name, local.dims, local.ty, &init_value)?;
        state.insert(local.name.to_owned(), init_value);
        assert_refinement(local.name, local.refinement, &state)?;
        timeline.push(format!("local {}", local.name), Some(span.clone()), &state);

        state = self.trace_stmt(local.body, state, tapes, timeline)?;
        assert_refinement(local.name, local.refinement, &state)?;

        let actual = state
            .get(local.name)
            .cloned()
            .ok_or_else(|| RuntimeError::new(format!("local `{}` disappeared", local.name)))?;
        let expected = eval(local.delocal, &state)?;
        validate_local_value(local.name, local.dims, local.ty, &actual)?;
        validate_local_value(local.name, local.dims, local.ty, &expected)?;
        validate_local_value(local.name, local.delocal_dims, local.delocal_ty, &actual)?;
        validate_local_value(local.name, local.delocal_dims, local.delocal_ty, &expected)?;
        if actual != expected {
            return Err(RuntimeError::at(
                format!(
                    "delocal `{}` expected {expected}, found {actual}",
                    local.name
                ),
                local.delocal.span.clone(),
            ));
        }
        assert_refinement(local.name, local.delocal_refinement, &state)?;

        state.remove(local.name);
        timeline.push(format!("delocal {}", local.name), Some(span), &state);
        Ok(state)
    }

    fn trace_iterate(
        &self,
        iterate: IterateRuntime<'_>,
        span: Range<usize>,
        mut state: State,
        tapes: &mut Tapes<'_>,
        timeline: &mut TimelineBuilder,
    ) -> Result<State, RuntimeError> {
        let bounds = eval_iterate_bounds(iterate, &state, span.clone())?;
        if state.contains(iterate.name) {
            return Err(RuntimeError::new(format!(
                "iterate variable `{}` would shadow an existing variable",
                iterate.name
            )));
        }

        state.insert(iterate.name.to_owned(), Value::Int(bounds.start));
        timeline.push(
            format!("iterate {}", iterate.name),
            Some(span.clone()),
            &state,
        );
        let mut current = bounds.start;
        while iterate_contains(current, bounds.end, bounds.step) {
            state.insert(iterate.name.to_owned(), Value::Int(current));
            state = self.trace_stmt(iterate.body, state, tapes, timeline)?;
            assert_iterate_variable(iterate.name, current, &state, span.clone())?;
            let Some(next) = next_iterate_value(current, bounds.end, bounds.step, span.clone())?
            else {
                break;
            };
            current = next;
            timeline.push(
                format!("iterate {} next", iterate.name),
                Some(span.clone()),
                &state,
            );
        }
        state.remove(iterate.name);
        timeline.push(format!("end {}", iterate.name), Some(span), &state);
        Ok(state)
    }
}

fn update_label(op: UpdateOp) -> &'static str {
    match op {
        UpdateOp::Add => "+=",
        UpdateOp::Sub => "-=",
        UpdateOp::Xor => "^=",
    }
}

struct LocalRuntime<'a> {
    name: &'a str,
    ty: Option<&'a TypeExpr>,
    dims: &'a [Option<usize>],
    refinement: Option<&'a SpannedExpr>,
    init: &'a SpannedExpr,
    body: &'a SpannedStmt,
    delocal_name: &'a str,
    delocal_ty: Option<&'a TypeExpr>,
    delocal_dims: &'a [Option<usize>],
    delocal_refinement: Option<&'a SpannedExpr>,
    delocal: &'a SpannedExpr,
}

struct ProcedureTrace<'a> {
    name: &'a str,
    args: &'a [Place],
    call_span: Range<usize>,
    direction: Direction,
}

#[derive(Debug, Clone, Copy)]
struct IterateBounds {
    start: i64,
    step: i64,
    end: i64,
}

#[derive(Debug, Clone, Copy)]
struct IterateRuntime<'a> {
    name: &'a str,
    start: &'a SpannedExpr,
    step: &'a SpannedExpr,
    end: &'a SpannedExpr,
    body: &'a SpannedStmt,
}

fn eval_iterate_bounds(
    iterate: IterateRuntime<'_>,
    state: &State,
    span: Range<usize>,
) -> Result<IterateBounds, RuntimeError> {
    let start = eval_int(iterate.start, state)?;
    let step = eval_int(iterate.step, state)?;
    let end = eval_int(iterate.end, state)?;
    if step == 0 {
        return Err(RuntimeError::at("iterate step cannot be zero", span));
    }
    Ok(IterateBounds { start, step, end })
}

fn iterate_contains(current: i64, end: i64, step: i64) -> bool {
    if step > 0 {
        current <= end
    } else {
        current >= end
    }
}

fn next_iterate_value(
    current: i64,
    end: i64,
    step: i64,
    span: Range<usize>,
) -> Result<Option<i64>, RuntimeError> {
    if current == end {
        return Ok(None);
    }

    let next = i128::from(current) + i128::from(step);
    if next < i128::from(i64::MIN) || next > i128::from(i64::MAX) {
        return Err(RuntimeError::at("iterate counter overflowed i64", span));
    }
    Ok(Some(next as i64))
}

fn assert_iterate_variable(
    name: &str,
    expected: i64,
    state: &State,
    span: Range<usize>,
) -> Result<(), RuntimeError> {
    let actual = state.get(name).ok_or_else(|| {
        RuntimeError::at(
            format!("iterate variable `{name}` disappeared"),
            span.clone(),
        )
    })?;
    match actual {
        Value::Int(actual) if *actual == expected => Ok(()),
        Value::Int(actual) => Err(RuntimeError::at(
            format!("iterate variable `{name}` expected {expected}, found {actual}"),
            span,
        )),
        other => Err(RuntimeError::at(
            format!("iterate variable `{name}` expected int {expected}, found {other}"),
            span,
        )),
    }
}

fn validate_declared_value(
    name: &str,
    dims: &[usize],
    ty: Option<&TypeExpr>,
    value: &Value,
) -> Result<(), RuntimeError> {
    if let Some((len, rest)) = dims.split_first() {
        let element_ty = match ty {
            Some(TypeExpr::Array { element }) => Some(&element.node),
            other => other,
        };
        match value {
            Value::Array(values) if values.len() == *len => {
                for element in values {
                    validate_declared_value(name, rest, element_ty, element)?;
                }
                return Ok(());
            }
            Value::Array(values) => {
                return Err(RuntimeError::new(format!(
                    "declaration `{name}` expected array length {len}, found {}",
                    values.len()
                )));
            }
            other => {
                return Err(RuntimeError::new(format!(
                    "declaration `{name}` expected array length {len}, found {other}"
                )));
            }
        }
    }

    match (ty, value) {
        (Some(TypeExpr::Bool), Value::Bool(_))
        | (Some(TypeExpr::Stack), Value::Stack(_))
        | (Some(TypeExpr::Int { .. }), Value::Int(_))
        | (None, Value::Int(_)) => Ok(()),
        (Some(TypeExpr::Array { element }), Value::Array(values)) => {
            for element_value in values {
                validate_declared_value(name, &[], Some(&element.node), element_value)?;
            }
            validate_rectangular_declared_array(name, values)?;
            Ok(())
        }
        (Some(TypeExpr::Bool), other) => Err(RuntimeError::new(format!(
            "declaration `{name}` expected bool, found {other}"
        ))),
        (Some(TypeExpr::Stack), other) => Err(RuntimeError::new(format!(
            "declaration `{name}` expected stack, found {other}"
        ))),
        (Some(TypeExpr::Array { .. }), other) => Err(RuntimeError::new(format!(
            "declaration `{name}` expected array, found {other}"
        ))),
        (Some(TypeExpr::Int { .. }) | None, other) => Err(RuntimeError::new(format!(
            "declaration `{name}` expected int, found {other}"
        ))),
    }
}

fn validate_rectangular_declared_array(name: &str, values: &[Value]) -> Result<(), RuntimeError> {
    let Some(Value::Array(first)) = values.first() else {
        return Ok(());
    };

    for value in &values[1..] {
        let Value::Array(row) = value else {
            continue;
        };
        if row.len() != first.len() {
            return Err(RuntimeError::new(format!(
                "declaration `{name}` has ragged array rows: expected length {}, found {}",
                first.len(),
                row.len()
            )));
        }
    }

    Ok(())
}

fn validate_local_value(
    name: &str,
    dims: &[Option<usize>],
    ty: Option<&TypeExpr>,
    value: &Value,
) -> Result<(), RuntimeError> {
    validate_local_dims(name, dims, value)?;
    if let Some(ty) = ty {
        validate_declared_value(name, &[], Some(ty), value)?;
    }
    Ok(())
}

fn validate_local_dims(
    name: &str,
    dims: &[Option<usize>],
    value: &Value,
) -> Result<(), RuntimeError> {
    let Some((len, rest)) = dims.split_first() else {
        return Ok(());
    };

    let Value::Array(values) = value else {
        return Err(RuntimeError::new(format!(
            "local `{name}` expected array, found {value}"
        )));
    };

    if let Some(len) = len
        && values.len() != *len
    {
        return Err(RuntimeError::new(format!(
            "local `{name}` expected array length {len}, found {}",
            values.len()
        )));
    }

    for value in values {
        validate_local_dims(name, rest, value)?;
    }

    Ok(())
}

fn declare_value(
    name: &str,
    ty: TypeExpr,
    dims: &[usize],
    init: Option<&SpannedExpr>,
    state: &mut State,
) -> Result<(), RuntimeError> {
    let value = if let Some(init) = init {
        eval(init, state)?
    } else {
        default_value(dims, Some(&ty))?
    };
    validate_declared_shape(name, dims, &value)?;

    if let Some(existing) = state.get(name) {
        validate_declared_value(name, dims, Some(&ty), existing)?;
        if existing == &value {
            return Ok(());
        }

        return Err(RuntimeError::new(format!(
            "declaration `{name}` expected {value}, found {existing}"
        )));
    }

    state.insert(name.to_owned(), value);
    Ok(())
}

fn assert_refinement(
    name: &str,
    refinement: Option<&SpannedExpr>,
    state: &State,
) -> Result<(), RuntimeError> {
    let Some(refinement) = refinement else {
        return Ok(());
    };

    let holds = eval_bool(refinement, state)?;
    if holds {
        Ok(())
    } else {
        Err(RuntimeError::at(
            format!("refinement for `{name}` evaluated to false"),
            refinement.span.clone(),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Call,
    Uncall,
}

impl Direction {
    fn verb(self) -> &'static str {
        match self {
            Self::Call => "call",
            Self::Uncall => "uncall",
        }
    }

    fn gerund(self) -> &'static str {
        match self {
            Self::Call => "calling",
            Self::Uncall => "uncalling",
        }
    }
}

enum FrozenCallArgs<'a> {
    Borrowed(&'a [Place]),
    Owned(Vec<Place>),
}

impl<'a> FrozenCallArgs<'a> {
    fn as_slice(&self) -> &[Place] {
        match self {
            Self::Borrowed(args) => args,
            Self::Owned(args) => args,
        }
    }
}

fn freeze_call_args<'a>(
    args: &'a [Place],
    state: &State,
) -> Result<FrozenCallArgs<'a>, RuntimeError> {
    if args.iter().all(|arg| arg.indices.is_empty()) {
        return Ok(FrozenCallArgs::Borrowed(args));
    }

    args.iter()
        .map(|arg| freeze_call_arg(arg, state))
        .collect::<Result<Vec<_>, _>>()
        .map(FrozenCallArgs::Owned)
}

fn freeze_call_arg(arg: &Place, state: &State) -> Result<Place, RuntimeError> {
    if arg.indices.is_empty() {
        return Ok(arg.clone());
    }

    let mut indices = Vec::with_capacity(arg.indices.len());
    for index in &arg.indices {
        let value = eval_int(index, state)?;
        if value < 0 {
            return Err(RuntimeError::at(
                format!("array index must be non-negative, found {value}"),
                index.span.clone(),
            ));
        }
        indices.push(Spanned::new(
            Expr::Int { value, unit: None },
            index.span.clone(),
        ));
    }

    Ok(Place::with_indices(arg.name.clone(), indices))
}

fn ensure_unique_args(name: &str, args: &[Place]) -> Result<(), RuntimeError> {
    for (index, arg) in args.iter().enumerate() {
        for other in args.iter().skip(index + 1) {
            if arg.name == other.name && (arg.is_indexed() ^ other.is_indexed()) {
                return Err(RuntimeError::new(format!(
                    "procedure `{name}` cannot mix whole-place and element arguments rooted at `{}`",
                    arg.name
                )));
            }

            if same_call_arg(arg, other) {
                let key = call_arg_key(arg);
                return Err(RuntimeError::new(format!(
                    "procedure `{name}` cannot be called with duplicate argument `{key}`"
                )));
            }
        }
    }

    Ok(())
}

fn same_call_arg(left: &Place, right: &Place) -> bool {
    left.name == right.name
        && left.indices.len() == right.indices.len()
        && left
            .indices
            .iter()
            .zip(&right.indices)
            .all(|(left, right)| match (&left.node, &right.node) {
                (Expr::Int { value: left, .. }, Expr::Int { value: right, .. }) => left == right,
                _ => false,
            })
}

fn call_arg_key(arg: &Place) -> String {
    if arg.indices.is_empty() {
        return arg.name.clone();
    }

    let suffix = arg
        .indices
        .iter()
        .map(|index| match &index.node {
            Expr::Int { value, .. } => format!("[{value}]"),
            _ => "[...]".to_owned(),
        })
        .collect::<String>();
    format!("{}{suffix}", arg.name)
}

fn assert_condition(
    expr: &SpannedExpr,
    state: &State,
    expected: bool,
    label: &str,
) -> Result<(), RuntimeError> {
    let actual = eval_bool(expr, state)?;
    if actual == expected {
        Ok(())
    } else {
        Err(RuntimeError::at(
            format!("{label} expected {expected}, found {actual}"),
            expr.span.clone(),
        ))
    }
}

fn eval(expr: &SpannedExpr, state: &State) -> Result<Value, RuntimeError> {
    match &expr.node {
        Expr::Int { value, .. } => Ok(Value::Int(*value)),
        Expr::Bool(value) => Ok(Value::Bool(*value)),
        Expr::Array(elements) => elements
            .iter()
            .map(|element| eval(element, state))
            .collect::<Result<Vec<_>, _>>()
            .and_then(array_value),
        Expr::Nil => Ok(Value::Stack(Vec::new())),
        Expr::Var(name) => state.get(name).cloned().ok_or_else(|| {
            RuntimeError::at(format!("undefined variable `{name}`"), expr.span.clone())
        }),
        Expr::Index { target, indices } => {
            eval_place(&Place::with_indices(target.clone(), indices.clone()), state)
        }
        Expr::Empty { target } => eval_empty(target, expr.span.clone(), state),
        Expr::Top { target } => eval_top(target, expr.span.clone(), state),
        Expr::Size { target } => eval_size(target, expr.span.clone(), state),
        Expr::Unary { op, expr } => eval_unary(*op, expr, state),
        Expr::Binary { op, left, right } => eval_binary(*op, left, right, state),
    }
}

fn eval_empty(target: &str, span: Range<usize>, state: &State) -> Result<Value, RuntimeError> {
    let stack = expect_stack(target, span, state)?;
    Ok(Value::Bool(stack.is_empty()))
}

fn eval_top(target: &str, span: Range<usize>, state: &State) -> Result<Value, RuntimeError> {
    let stack = expect_stack(target, span.clone(), state)?;
    stack
        .first()
        .copied()
        .map(Value::Int)
        .ok_or_else(|| RuntimeError::at(format!("top({target}) expected a non-empty stack"), span))
}

fn eval_size(target: &str, span: Range<usize>, state: &State) -> Result<Value, RuntimeError> {
    let value = state
        .get(target)
        .ok_or_else(|| RuntimeError::at(format!("undefined variable `{target}`"), span.clone()))?;
    match value {
        Value::Array(values) => Ok(Value::Int(i64::try_from(values.len()).map_err(|_| {
            RuntimeError::at(format!("array `{target}` length exceeds i64"), span.clone())
        })?)),
        Value::Stack(values) => Ok(Value::Int(i64::try_from(values.len()).map_err(|_| {
            RuntimeError::at(format!("stack `{target}` length exceeds i64"), span.clone())
        })?)),
        other => Err(RuntimeError::at(
            format!("expected `{target}` to be an array or stack, found {other}"),
            span,
        )),
    }
}

fn array_value(values: Vec<Value>) -> Result<Value, RuntimeError> {
    validate_rectangular_runtime_array("array literal", &values)?;
    Ok(Value::Array(values))
}

fn validate_rectangular_runtime_array(context: &str, values: &[Value]) -> Result<(), RuntimeError> {
    let Some(Value::Array(first)) = values.first() else {
        return Ok(());
    };

    for value in &values[1..] {
        let Value::Array(row) = value else {
            continue;
        };
        if row.len() != first.len() {
            return Err(RuntimeError::new(format!(
                "{context} has ragged rows: expected length {}, found {}",
                first.len(),
                row.len()
            )));
        }
    }

    Ok(())
}

fn eval_unary(op: UnaryOp, expr: &SpannedExpr, state: &State) -> Result<Value, RuntimeError> {
    Ok(match op {
        UnaryOp::Neg => Value::Int(eval_int(expr, state)?.wrapping_neg()),
        UnaryOp::Not => Value::Bool(!eval_bool(expr, state)?),
    })
}

fn eval_binary(
    op: BinaryOp,
    left: &SpannedExpr,
    right: &SpannedExpr,
    state: &State,
) -> Result<Value, RuntimeError> {
    Ok(match op {
        BinaryOp::Add => Value::Int(eval_int(left, state)?.wrapping_add(eval_int(right, state)?)),
        BinaryOp::Sub => Value::Int(eval_int(left, state)?.wrapping_sub(eval_int(right, state)?)),
        BinaryOp::Mul => Value::Int(eval_int(left, state)?.wrapping_mul(eval_int(right, state)?)),
        BinaryOp::FixedMul => Value::Int(fixed_mul_q31(
            eval_int(left, state)?,
            eval_int(right, state)?,
        )),
        BinaryOp::Div => {
            let divisor = eval_int(right, state)?;
            if divisor == 0 {
                return Err(RuntimeError::at("division by zero", right.span.clone()));
            }
            Value::Int(eval_int(left, state)?.wrapping_div(divisor))
        }
        BinaryOp::Rem => {
            let divisor = eval_int(right, state)?;
            if divisor == 0 {
                return Err(RuntimeError::at("remainder by zero", right.span.clone()));
            }
            Value::Int(eval_int(left, state)?.wrapping_rem(divisor))
        }
        BinaryOp::Shl => {
            Value::Int(eval_int(left, state)?.wrapping_shl(eval_int(right, state)? as u32))
        }
        BinaryOp::Shr => {
            Value::Int(eval_int(left, state)?.wrapping_shr(eval_int(right, state)? as u32))
        }
        BinaryOp::BitAnd => eval_bitwise(left, right, state, "bitwise and", |left, right| {
            left & right
        })?,
        BinaryOp::BitXor => eval_bitwise(left, right, state, "bitwise xor", |left, right| {
            left ^ right
        })?,
        BinaryOp::BitOr => {
            eval_bitwise(left, right, state, "bitwise or", |left, right| left | right)?
        }
        BinaryOp::Eq => Value::Bool(eval(left, state)? == eval(right, state)?),
        BinaryOp::NotEq => Value::Bool(eval(left, state)? != eval(right, state)?),
        BinaryOp::Lt => Value::Bool(eval_int(left, state)? < eval_int(right, state)?),
        BinaryOp::LtEq => Value::Bool(eval_int(left, state)? <= eval_int(right, state)?),
        BinaryOp::Gt => Value::Bool(eval_int(left, state)? > eval_int(right, state)?),
        BinaryOp::GtEq => Value::Bool(eval_int(left, state)? >= eval_int(right, state)?),
        BinaryOp::And => Value::Bool(eval_bool(left, state)? && eval_bool(right, state)?),
        BinaryOp::Or => Value::Bool(eval_bool(left, state)? || eval_bool(right, state)?),
    })
}

fn fixed_mul_q31(left: i64, right: i64) -> i64 {
    ((i128::from(left) * i128::from(right)) >> 31) as i64
}

fn eval_bitwise(
    left: &SpannedExpr,
    right: &SpannedExpr,
    state: &State,
    op_name: &str,
    op: impl FnOnce(i64, i64) -> i64,
) -> Result<Value, RuntimeError> {
    let span = left.span.start..right.span.end;
    match (eval(left, state)?, eval(right, state)?) {
        (Value::Bool(left), Value::Bool(right)) => Ok(Value::Bool(op_bool(op_name, left, right))),
        (Value::Int(left), Value::Int(right)) => Ok(Value::Int(op(left, right))),
        (left, right) => Err(RuntimeError::at(
            format!(
                "expected matching int or bool operands for {op_name}, found {left} and {right}"
            ),
            span,
        )),
    }
}

fn op_bool(op_name: &str, left: bool, right: bool) -> bool {
    match op_name {
        "bitwise and" => left & right,
        "bitwise xor" => left ^ right,
        "bitwise or" => left | right,
        _ => unreachable!("unsupported boolean bitwise operator"),
    }
}

fn eval_int(expr: &SpannedExpr, state: &State) -> Result<i64, RuntimeError> {
    match eval(expr, state)? {
        Value::Int(value) => Ok(value),
        Value::Bool(value) => Err(RuntimeError::at(
            format!("expected int, found bool `{value}`"),
            expr.span.clone(),
        )),
        Value::Array(value) => Err(RuntimeError::at(
            format!("expected int, found array with {} element(s)", value.len()),
            expr.span.clone(),
        )),
        Value::Stack(value) => Err(RuntimeError::at(
            format!("expected int, found stack with {} element(s)", value.len()),
            expr.span.clone(),
        )),
    }
}

fn eval_bool(expr: &SpannedExpr, state: &State) -> Result<bool, RuntimeError> {
    match eval(expr, state)? {
        Value::Bool(value) => Ok(value),
        Value::Int(value) => Ok(value != 0),
        Value::Array(value) => Err(RuntimeError::at(
            format!("expected bool, found array with {} element(s)", value.len()),
            expr.span.clone(),
        )),
        Value::Stack(value) => Err(RuntimeError::at(
            format!("expected bool, found stack with {} element(s)", value.len()),
            expr.span.clone(),
        )),
    }
}

fn updated_value(
    target: &Place,
    op: UpdateOp,
    expr: &SpannedExpr,
    state: &State,
) -> Result<Value, RuntimeError> {
    match op {
        UpdateOp::Add => {
            let current = expect_place_int(target, state)?;
            let rhs = eval_int(expr, state)?;
            Ok(Value::Int(current.wrapping_add(rhs)))
        }
        UpdateOp::Sub => {
            let current = expect_place_int(target, state)?;
            let rhs = eval_int(expr, state)?;
            Ok(Value::Int(current.wrapping_sub(rhs)))
        }
        UpdateOp::Xor => match (eval_place(target, state)?, eval(expr, state)?) {
            (Value::Int(current), Value::Int(rhs)) => Ok(Value::Int(current ^ rhs)),
            (Value::Bool(current), Value::Bool(rhs)) => Ok(Value::Bool(current ^ rhs)),
            (current, rhs) => Err(RuntimeError::at(
                format!(
                    "expected matching int or bool operands for xor update, found {current} and {rhs}"
                ),
                expr.span.clone(),
            )),
        },
    }
}

fn expect_place_int(place: &Place, state: &State) -> Result<i64, RuntimeError> {
    match eval_place(place, state)? {
        Value::Int(value) => Ok(value),
        Value::Bool(value) => Err(RuntimeError::new(format!(
            "expected `{}` to be int, found bool `{value}`",
            display_place(place)
        ))),
        Value::Array(value) => Err(RuntimeError::new(format!(
            "expected `{}` to be int, found array with {} element(s)",
            display_place(place),
            value.len()
        ))),
        Value::Stack(value) => Err(RuntimeError::new(format!(
            "expected `{}` to be int, found stack with {} element(s)",
            display_place(place),
            value.len()
        ))),
    }
}

fn push_stack(source: &Place, stack: &str, state: &mut State) -> Result<(), RuntimeError> {
    let value = expect_place_int(source, state)?;
    let mut stack_value = expect_stack(stack, 0..0, state)?.to_vec();
    stack_value.insert(0, value);
    assign_place(source, state, Value::Int(0))?;
    state.insert(stack.to_owned(), Value::Stack(stack_value));
    Ok(())
}

fn pop_stack(target: &Place, stack: &str, state: &mut State) -> Result<(), RuntimeError> {
    let current = expect_place_int(target, state)?;
    if current != 0 {
        return Err(RuntimeError::new(format!(
            "pop({}, {stack}) expected target to be zero, found {current}",
            display_place(target)
        )));
    }

    let mut stack_value = expect_stack(stack, 0..0, state)?.to_vec();
    let value = if stack_value.is_empty() {
        return Err(RuntimeError::new(format!(
            "pop expected non-empty stack `{stack}`"
        )));
    } else {
        stack_value.remove(0)
    };
    state.insert(stack.to_owned(), Value::Stack(stack_value));
    assign_place(target, state, Value::Int(value))
}

fn expect_stack<'a>(
    name: &str,
    span: Range<usize>,
    state: &'a State,
) -> Result<&'a [i64], RuntimeError> {
    let value = state
        .get(name)
        .ok_or_else(|| RuntimeError::at(format!("undefined variable `{name}`"), span.clone()))?;
    match value {
        Value::Stack(values) => Ok(values),
        other => Err(RuntimeError::at(
            format!("expected `{name}` to be a stack, found {other}"),
            span,
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ResolvedPlace {
    Whole(String),
    Element { name: String, indices: Vec<usize> },
}

impl ResolvedPlace {
    fn aliases(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Whole(left), Self::Whole(right)) => left == right,
            (Self::Whole(left), Self::Element { name: right, .. })
            | (Self::Element { name: left, .. }, Self::Whole(right)) => left == right,
            (
                Self::Element {
                    name: left_name,
                    indices: left_indices,
                },
                Self::Element {
                    name: right_name,
                    indices: right_indices,
                },
            ) => left_name == right_name && left_indices == right_indices,
        }
    }
}

fn ensure_update_rhs_does_not_read_target(
    target: &Place,
    expr: &SpannedExpr,
    state: &State,
) -> Result<(), RuntimeError> {
    let target = resolve_place(target, state)?;
    let mut reads = Vec::new();
    collect_read_places(expr, state, &mut reads)?;
    if reads.iter().any(|read| target.aliases(read)) {
        return Err(RuntimeError::at(
            "irreversible update: right-hand side reads the value being changed",
            expr.span.clone(),
        ));
    }

    Ok(())
}

fn collect_read_places(
    expr: &SpannedExpr,
    state: &State,
    reads: &mut Vec<ResolvedPlace>,
) -> Result<(), RuntimeError> {
    match &expr.node {
        Expr::Int { .. } | Expr::Bool(_) | Expr::Nil | Expr::Size { .. } => {}
        Expr::Array(elements) => {
            for element in elements {
                collect_read_places(element, state, reads)?;
            }
        }
        Expr::Var(name) => reads.push(ResolvedPlace::Whole(name.clone())),
        Expr::Index { target, indices } => {
            for index in indices {
                collect_read_places(index, state, reads)?;
            }
            reads.push(resolve_place(
                &Place::with_indices(target.clone(), indices.clone()),
                state,
            )?);
        }
        Expr::Empty { target } | Expr::Top { target } => {
            reads.push(ResolvedPlace::Whole(target.clone()));
        }
        Expr::Unary { expr, .. } => collect_read_places(expr, state, reads)?,
        Expr::Binary { left, right, .. } => {
            collect_read_places(left, state, reads)?;
            collect_read_places(right, state, reads)?;
        }
    }

    Ok(())
}

fn resolve_place(place: &Place, state: &State) -> Result<ResolvedPlace, RuntimeError> {
    if place.indices.is_empty() {
        return Ok(ResolvedPlace::Whole(place.name.clone()));
    }

    let indices = checked_indices(&place.indices, state)?
        .into_iter()
        .map(|(index, _)| index)
        .collect();
    Ok(ResolvedPlace::Element {
        name: place.name.clone(),
        indices,
    })
}

fn eval_place(place: &Place, state: &State) -> Result<Value, RuntimeError> {
    if place.indices.is_empty() {
        return state
            .get(&place.name)
            .cloned()
            .ok_or_else(|| RuntimeError::new(format!("undefined variable `{}`", place.name)));
    }

    let indices = checked_indices(&place.indices, state)?;
    let mut current = state
        .get(&place.name)
        .ok_or_else(|| RuntimeError::new(format!("undefined variable `{}`", place.name)))?;

    for (index, index_span) in indices {
        current = match current {
            Value::Array(values) => values.get(index).ok_or_else(|| {
                RuntimeError::at(
                    format!(
                        "array `{}` index {index} out of bounds for length {}",
                        place.name,
                        values.len()
                    ),
                    index_span,
                )
            })?,
            other => {
                return Err(RuntimeError::new(format!(
                    "expected `{}` to be an array, found {other}",
                    place.name
                )));
            }
        };
    }

    Ok(current.clone())
}

fn assign_place(place: &Place, state: &mut State, value: Value) -> Result<(), RuntimeError> {
    if place.indices.is_empty() {
        state.insert(place.name.clone(), value);
        return Ok(());
    }

    let indices = checked_indices(&place.indices, state)?;
    let root = state
        .store
        .get_mut(&place.name)
        .ok_or_else(|| RuntimeError::new(format!("undefined variable `{}`", place.name)))?;

    assign_indexed_place(&place.name, root, &indices, value)
}

fn assign_indexed_place(
    name: &str,
    root: &mut Value,
    indices: &[(usize, Range<usize>)],
    value: Value,
) -> Result<(), RuntimeError> {
    let Some(((index, index_span), rest)) = indices.split_first() else {
        *root = value;
        return Ok(());
    };

    match root {
        Value::Array(values) => {
            let len = values.len();
            let slot = values.get_mut(*index).ok_or_else(|| {
                RuntimeError::at(
                    format!("array `{name}` index {index} out of bounds for length {len}"),
                    index_span.clone(),
                )
            })?;
            assign_indexed_place(name, slot, rest, value)
        }
        other => Err(RuntimeError::new(format!(
            "expected `{name}` to be an array, found {other}"
        ))),
    }
}

fn checked_indices(
    indices: &[SpannedExpr],
    state: &State,
) -> Result<Vec<(usize, Range<usize>)>, RuntimeError> {
    indices
        .iter()
        .map(|index| checked_index(index, state).map(|value| (value, index.span.clone())))
        .collect()
}

fn checked_index(index: &SpannedExpr, state: &State) -> Result<usize, RuntimeError> {
    let value = eval_int(index, state)?;
    usize::try_from(value).map_err(|_| {
        RuntimeError::at(
            format!("array index must be non-negative, found {value}"),
            index.span.clone(),
        )
    })
}

fn display_place(place: &Place) -> String {
    if place.is_indexed() {
        format!("{}{}", place.name, "[...]".repeat(place.indices.len()))
    } else {
        place.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use reverie_syntax::parse_program;
    use reverie_syntax::{Spanned, UpdateOp};

    use super::*;

    const GLOBALS_SOURCE: &str = r#"
global x;
global xs[3];

proc bump_globals() {
  x += 1;
  xs[0] += 1;
  xs[1] += 2;
  xs[2] += 1
}

call bump_globals()
"#;

    const IO_SOURCE: &str = r#"
global x;

write x;
read x;
write x
"#;

    const STACK_SOURCE: &str = r#"
local s: stack = nil
  local x = 1
    push(x, s);
    pop(x, s)
  delocal x = 1
delocal s = nil
"#;

    const FIB_SOURCE: &str = r#"
from i == 0 do
  a += b;
  a <=> b;
  i += 1
loop
  skip
until i == n
"#;

    const PROC_SOURCE: &str = r#"
proc bump(x) {
  local t = 0
    t += x;
    x += 1
  delocal t = x - 1
}

call bump(n)
"#;

    const ARRAY_SOURCE: &str = "xs[1] += delta; xs[0] <=> xs[2]";

    const NEGATION_SOURCE: &str = r#"
local x = 0
  local y = -1
    x += -y
  delocal y = -1
delocal x = 1
"#;

    const JANUS_OPERATORS_SOURCE: &str = r#"
local x = 5
  local y = 3
    x ^= y;
    x ^= y;
    y += (x & 1) << 1;
    y -= 2
  delocal y = 3
delocal x = 5
"#;

    const JANUS_OPTIONAL_CONTROL_SOURCE: &str = r#"
local flag = true
  local x = 0
    if flag then
      x += 1
    fi x == 1
  delocal x = 1
delocal flag = true
"#;

    const JANUS_PROCEDURE_SYNTAX_SOURCE: &str = r#"
procedure main()
  int n = 1
  n += 1
"#;

    const SIZE_SOURCE: &str = r#"
local xs: array<int> = [1, 2, 3]
  assert size(xs) == 3
delocal xs = [1, 2, 3]
"#;

    const ASSERT_SOURCE: &str = r#"
assert x == 0;
target += 1;
assert target == 3;
target -= 1
"#;

    const REFINEMENT_SOURCE: &str = r#"
proc round_trip(n: int where n >= 0) {
  n += 1;
  n -= 1
}

local n: int where n >= 0 = 0
  call round_trip(n)
delocal n = 0
"#;

    const PROC_RUNTIME_ERROR_SOURCE: &str = r#"
proc boom(x) {
  x += 1 / 0
}

call boom(n)
"#;

    const REFINEMENT_VIOLATION_SOURCE: &str = r#"
local n: int where n >= 0 = 0
  n -= 1
delocal n = -1
"#;

    #[test]
    fn skip_leaves_state_empty() {
        let program = parse_program("skip").expect("program parses");
        let state = run_program(&program).expect("program runs");

        assert_eq!(state, State::empty());
        assert_eq!(state.to_string(), "{}");
    }

    #[test]
    fn sequence_of_skips_leaves_state_empty() {
        let program = parse_program("skip; skip").expect("program parses");
        let state = run_program(&program).expect("program runs");

        assert_eq!(state, State::empty());
    }

    #[test]
    fn standalone_assert_checks_a_bool_without_changing_state() {
        let program = parse_program("assert !(x != 1)").expect("program parses");
        let initial = State::from_bindings([("x".to_owned(), Value::Int(1))]);

        let tree = execute(&program, initial.clone()).expect("tree runs");
        let slot = execute_compiled(&program, initial.clone()).expect("slot runs");

        assert_eq!(tree, initial);
        assert_eq!(slot, initial);
    }

    #[test]
    fn bitwise_and_shift_expressions_run_in_both_engines() {
        let program = parse_program("x += (mask & 3) | (1 << 4); y += x >> 1; q += one */ half")
            .expect("program parses");
        let initial = State::from_bindings([
            ("half".to_owned(), Value::Int(1_073_741_824)),
            ("mask".to_owned(), Value::Int(5)),
            ("one".to_owned(), Value::Int(2_147_483_648)),
            ("q".to_owned(), Value::Int(0)),
            ("x".to_owned(), Value::Int(0)),
            ("y".to_owned(), Value::Int(0)),
        ]);
        let expected = State::from_bindings([
            ("half".to_owned(), Value::Int(1_073_741_824)),
            ("mask".to_owned(), Value::Int(5)),
            ("one".to_owned(), Value::Int(2_147_483_648)),
            ("q".to_owned(), Value::Int(1_073_741_824)),
            ("x".to_owned(), Value::Int(17)),
            ("y".to_owned(), Value::Int(8)),
        ]);

        let tree = execute(&program, initial.clone()).expect("tree runs");
        let slot = execute_compiled(&program, initial).expect("slot runs");

        assert_eq!(tree, expected);
        assert_eq!(slot, expected);
    }

    #[test]
    fn boolean_xor_update_runs_in_both_engines() {
        let program = parse_program("flag ^= true").expect("program parses");
        let initial = State::from_bindings([("flag".to_owned(), Value::Bool(false))]);
        let expected = State::from_bindings([("flag".to_owned(), Value::Bool(true))]);

        let tree = execute(&program, initial.clone()).expect("tree runs");
        let slot = execute_compiled(&program, initial.clone()).expect("slot runs");
        let reverse = execute_backward(&program, expected.clone()).expect("reverse runs");
        let compiled_reverse =
            execute_compiled_backward(&program, expected.clone()).expect("compiled reverse runs");

        assert_eq!(tree, expected);
        assert_eq!(slot, expected);
        assert_eq!(reverse, initial);
        assert_eq!(compiled_reverse, initial);
    }

    #[test]
    fn legacy_int_conditions_use_nonzero_truth() {
        let program = parse_program("if mask & 4 then x += 1 fi x == 1").expect("program parses");
        let initial = State::from_bindings([
            ("mask".to_owned(), Value::Int(5)),
            ("x".to_owned(), Value::Int(0)),
        ]);
        let expected = State::from_bindings([
            ("mask".to_owned(), Value::Int(5)),
            ("x".to_owned(), Value::Int(1)),
        ]);

        let tree = execute(&program, initial.clone()).expect("tree runs");
        let slot = execute_compiled(&program, initial).expect("slot runs");

        assert_eq!(tree, expected);
        assert_eq!(slot, expected);
    }

    #[test]
    fn standalone_assert_failure_has_runtime_span() {
        let program = parse_program("assert x == 1").expect("program parses");
        let initial = State::from_bindings([("x".to_owned(), Value::Int(0))]);
        let error = execute(&program, initial).expect_err("assertion fails");

        assert!(error.to_string().contains("assertion expected true"));
        assert_eq!(error.span(), Some(7..13));
    }

    #[test]
    fn updates_use_wrapping_i64_arithmetic() {
        let program = parse_program("x += 1").expect("program parses");
        let state = State::from_bindings([("x".to_owned(), Value::Int(i64::MAX))]);
        let state = execute(&program, state).expect("program runs");

        assert_eq!(state.get("x"), Some(&Value::Int(i64::MIN)));
    }

    #[test]
    fn unary_numeric_negation_uses_wrapping_i64_arithmetic() {
        let program = parse_program("x += -y").expect("program parses");
        let initial = State::from_bindings([
            ("x".to_owned(), Value::Int(0)),
            ("y".to_owned(), Value::Int(i64::MIN)),
        ]);

        let tree = execute(&program, initial.clone()).expect("tree runs");
        let slot = execute_compiled(&program, initial).expect("slot runs");

        assert_eq!(tree.get("x"), Some(&Value::Int(i64::MIN)));
        assert_eq!(slot, tree);
    }

    #[test]
    fn swap_exchanges_values() {
        let program = parse_program("x <=> y").expect("program parses");
        let state = State::from_bindings([
            ("x".to_owned(), Value::Int(1)),
            ("y".to_owned(), Value::Bool(true)),
        ]);
        let state = execute(&program, state).expect("program runs");

        assert_eq!(state.get("x"), Some(&Value::Bool(true)));
        assert_eq!(state.get("y"), Some(&Value::Int(1)));
    }

    #[test]
    fn array_element_updates_and_swaps_are_reversible() {
        let program = parse_program("xs[1] += delta; xs[0] <=> xs[2]").expect("program parses");
        let initial = State::from_bindings([
            (
                "xs".to_owned(),
                Value::Array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]),
            ),
            ("delta".to_owned(), Value::Int(5)),
        ]);

        let forward = execute(&program, initial.clone()).expect("program runs forward");
        assert_eq!(forward.to_string(), "{delta = 5, xs = [30, 25, 10]}");

        let backward = execute_backward(&program, forward).expect("program runs backward");
        assert_eq!(backward, initial);
    }

    #[test]
    fn multidimensional_array_elements_run_forward_and_backward() {
        let program =
            parse_program("xs[1][0] += delta; xs[0][1] <=> xs[1][1]").expect("program parses");
        let initial = State::from_bindings([
            (
                "xs".to_owned(),
                Value::Array(vec![
                    Value::Array(vec![Value::Int(1), Value::Int(2)]),
                    Value::Array(vec![Value::Int(3), Value::Int(4)]),
                ]),
            ),
            ("delta".to_owned(), Value::Int(5)),
        ]);

        let tree = execute(&program, initial.clone()).expect("tree engine runs");
        assert_eq!(tree.to_string(), "{delta = 5, xs = [[1, 4], [8, 2]]}");
        let tree_back = execute_backward(&program, tree.clone()).expect("tree engine reverses");
        assert_eq!(tree_back, initial);

        let slot = execute_compiled(&program, initial.clone()).expect("slot engine runs");
        assert_eq!(slot, tree);
        let slot_back = execute_compiled_backward(&program, slot).expect("slot engine reverses");
        assert_eq!(slot_back, initial);
    }

    #[test]
    fn indexed_update_rejects_runtime_self_alias() {
        let program = parse_program("xs[i] += xs[j]").expect("program parses");
        let initial = State::from_bindings([
            (
                "xs".to_owned(),
                Value::Array(vec![Value::Int(1), Value::Int(2)]),
            ),
            ("i".to_owned(), Value::Int(1)),
            ("j".to_owned(), Value::Int(1)),
        ]);

        let tree = execute(&program, initial.clone()).expect_err("tree rejects aliased update");
        assert!(tree.to_string().contains("reads the value being changed"));

        let slot = execute_compiled(&program, initial).expect_err("slot rejects aliased update");
        assert!(slot.to_string().contains("reads the value being changed"));
    }

    #[test]
    fn globals_are_zero_initialized_and_visible_in_parameterless_procedures() {
        let program = parse_program(GLOBALS_SOURCE).expect("parses");

        let tree = execute(&program, State::empty()).expect("tree engine runs");
        assert_eq!(tree.to_string(), "{x = 1, xs = [1, 2, 1]}");

        let slot = execute_compiled(&program, State::empty()).expect("slot engine runs");
        assert_eq!(slot, tree);

        let backward = execute_compiled_backward(&program, slot).expect("slot engine reverses");
        assert_eq!(backward.to_string(), "{x = 0, xs = [0, 0, 0]}");
    }

    #[test]
    fn seeded_globals_must_match_declared_array_shape() {
        let program = parse_program("global xs[3]; skip").expect("parses");
        let initial = State::from_bindings([(
            "xs".to_owned(),
            Value::Array(vec![Value::Int(1), Value::Int(2)]),
        )]);

        let tree = execute(&program, initial.clone()).expect_err("tree rejects short global seed");
        assert!(tree.to_string().contains("expected array length 3"));

        let slot = execute_compiled(&program, initial).expect_err("slot rejects short global seed");
        assert!(slot.to_string().contains("expected array length 3"));
    }

    #[test]
    fn typed_global_arrays_run_in_both_engines() {
        let program =
            parse_program("global flags[3]: bool = [true, false, true]; flags[1] ^= true")
                .expect("parses");

        let tree = execute(&program, State::empty()).expect("tree engine runs");
        assert_eq!(tree.to_string(), "{flags = [true, true, true]}");

        let slot = execute_compiled(&program, State::empty()).expect("slot engine runs");
        assert_eq!(slot, tree);

        let backward = execute_compiled_backward(&program, slot).expect("slot engine reverses");
        assert_eq!(backward.to_string(), "{flags = [true, false, true]}");
    }

    #[test]
    fn seeded_globals_must_match_declared_value_type() {
        let scalar_program = parse_program("global x; skip").expect("parses");
        let scalar_initial = State::from_bindings([("x".to_owned(), Value::Array(Vec::new()))]);

        let tree = execute(&scalar_program, scalar_initial.clone())
            .expect_err("tree rejects array scalar");
        assert!(tree.to_string().contains("expected int"));

        let slot = execute_compiled(&scalar_program, scalar_initial)
            .expect_err("slot rejects array scalar");
        assert!(slot.to_string().contains("expected int"));

        let array_program = parse_program("global xs[3]; skip").expect("parses");
        let array_initial = State::from_bindings([(
            "xs".to_owned(),
            Value::Array(vec![
                Value::Array(Vec::new()),
                Value::Array(Vec::new()),
                Value::Array(Vec::new()),
            ]),
        )]);

        let tree = execute(&array_program, array_initial.clone())
            .expect_err("tree rejects non-int array elements");
        assert!(tree.to_string().contains("expected int"));

        let slot = execute_compiled(&array_program, array_initial)
            .expect_err("slot rejects non-int array elements");
        assert!(slot.to_string().contains("expected int"));
    }

    #[test]
    fn array_typed_seeded_globals_must_be_rectangular() {
        let program = parse_program("global matrix: array<array<int>>; skip").expect("parses");
        let initial = State::from_bindings([(
            "matrix".to_owned(),
            Value::Array(vec![
                Value::Array(vec![Value::Int(1), Value::Int(2)]),
                Value::Array(vec![Value::Int(3)]),
            ]),
        )]);

        let tree = execute(&program, initial.clone()).expect_err("tree rejects ragged seed");
        assert!(tree.to_string().contains("has ragged array rows"));

        let slot = execute_compiled(&program, initial).expect_err("slot rejects ragged seed");
        assert!(slot.to_string().contains("has ragged array rows"));
    }

    #[test]
    fn seeded_declarations_report_type_mismatches_before_value_mismatches() {
        let program = parse_program("int x\nskip").expect("parses");
        let initial = State::from_bindings([("x".to_owned(), Value::Array(Vec::new()))]);

        let tree = execute(&program, initial.clone()).expect_err("tree rejects array scalar");
        assert!(tree.to_string().contains("expected int"));

        let slot = execute_compiled(&program, initial).expect_err("slot rejects array scalar");
        assert!(slot.to_string().contains("expected int"));
    }

    #[test]
    fn janus_main_declarations_initialize_program_store() {
        let program = parse_program(
            r#"
procedure main()
  int n = 3
  int xs[3] = {1, 2, 3}
  stack s
  assert size(xs) == n
  assert empty(s)
"#,
        )
        .expect("program parses");

        let tree = execute(&program, State::empty()).expect("tree engine runs");
        assert_eq!(tree.to_string(), "{n = 3, s = nil, xs = [1, 2, 3]}");

        let slot = execute_compiled(&program, State::empty()).expect("slot engine runs");
        assert_eq!(slot, tree);

        let backward = execute_compiled_backward(&program, slot).expect("slot engine reverses");
        assert_eq!(backward.to_string(), "{n = 3, s = nil, xs = [1, 2, 3]}");
    }

    #[test]
    fn multidimensional_source_declarations_initialize_nested_arrays() {
        let program = parse_program(
            r#"
procedure main()
  int xs[2][2]
  xs[1][0] += 7
"#,
        )
        .expect("program parses");

        let tree = execute(&program, State::empty()).expect("tree engine runs");
        assert_eq!(tree.to_string(), "{xs = [[0, 0], [7, 0]]}");

        let slot = execute_compiled(&program, State::empty()).expect("slot engine runs");
        assert_eq!(slot, tree);
    }

    #[test]
    fn tape_io_round_trips() {
        let program = parse_program(IO_SOURCE).expect("parses");
        let initial = IoState::new(State::empty(), [Value::Int(7)]);

        let tree = execute_io(&program, initial.clone()).expect("tree engine runs");
        assert_eq!(tree.store().to_string(), "{x = 7}");
        assert_eq!(
            tree.output(),
            &[Value::Int(0), Value::Int(0), Value::Int(7)]
        );

        let slot = execute_compiled_io(&program, initial).expect("slot engine runs");
        assert_eq!(slot, tree);

        let backward =
            execute_compiled_io_backward(&program, slot).expect("slot engine reverses I/O");
        assert_eq!(backward.store().to_string(), "{x = 0}");
        assert_eq!(
            backward.input().iter().collect::<Vec<_>>(),
            vec![&Value::Int(7)]
        );
        assert!(backward.output().is_empty());
    }

    #[test]
    fn local_array_literals_can_be_delocaled() {
        let program = parse_program(
            "local xs: array<int> = [1, 2, 3] xs[1] += 5; xs[0] <=> xs[2] delocal xs = [3, 7, 1]",
        )
        .expect("program parses");
        let state = execute(&program, State::empty()).expect("program runs");

        assert_eq!(state, State::empty());
    }

    #[test]
    fn typed_local_array_literals_must_be_rectangular_at_runtime() {
        let program = parse_program(
            "local matrix: array<array<int>> = [[1, 2], [3]] skip delocal matrix = [[1, 2], [3]]",
        )
        .expect("program parses");

        let tree = execute(&program, State::empty()).expect_err("tree rejects ragged local");
        assert!(tree.to_string().contains("array literal has ragged rows"));

        let slot =
            execute_compiled(&program, State::empty()).expect_err("slot rejects ragged local");
        assert!(slot.to_string().contains("array literal has ragged rows"));
    }

    #[test]
    fn untyped_array_literals_must_be_rectangular_at_runtime() {
        let program =
            parse_program("local matrix = [[1, 2], [3]] skip delocal matrix = [[1, 2], [3]]")
                .expect("program parses");

        let tree = execute(&program, State::empty()).expect_err("tree rejects ragged literal");
        assert!(tree.to_string().contains("array literal has ragged rows"));

        let slot =
            execute_compiled(&program, State::empty()).expect_err("slot rejects ragged literal");
        assert!(slot.to_string().contains("array literal has ragged rows"));
    }

    #[test]
    fn sized_type_first_local_arrays_run_in_both_engines() {
        let program =
            parse_program("local int xs[3] = [1, 2, 3] xs[1] += 5 delocal int xs[3] = [1, 7, 3]")
                .expect("program parses");

        let tree = execute(&program, State::empty()).expect("tree engine runs");
        assert_eq!(tree, State::empty());

        let slot = execute_compiled(&program, State::empty()).expect("slot engine runs");
        assert_eq!(slot, State::empty());
    }

    #[test]
    fn type_first_delocal_refinements_run_in_both_engines() {
        let program = parse_program("local int x = 0 x += 1 delocal int x where x > 0 = 1")
            .expect("program parses");

        let tree = execute(&program, State::empty()).expect("tree engine runs");
        assert_eq!(tree, State::empty());

        let slot = execute_compiled(&program, State::empty()).expect("slot engine runs");
        assert_eq!(slot, State::empty());
    }

    #[test]
    fn type_first_delocal_refinement_failures_run_in_both_engines() {
        let program = parse_program("local int x = 0 x += 1 delocal int x where x == 0 = 1")
            .expect("program parses");

        let tree = execute(&program, State::empty()).expect_err("tree engine rejects");
        assert!(
            tree.to_string()
                .contains("refinement for `x` evaluated to false")
        );

        let slot = execute_compiled(&program, State::empty()).expect_err("slot engine rejects");
        assert!(
            slot.to_string()
                .contains("refinement for `x` evaluated to false")
        );
    }

    #[test]
    fn reverie_style_delocal_refinements_run_in_both_engines() {
        let program = parse_program("local x: int = 0 x += 1 delocal x: int where x > 0 = 1")
            .expect("program parses");

        let tree = execute(&program, State::empty()).expect("tree engine runs");
        assert_eq!(tree, State::empty());

        let slot = execute_compiled(&program, State::empty()).expect("slot engine runs");
        assert_eq!(slot, State::empty());
    }

    #[test]
    fn stack_operations_run_forward_and_backward() {
        let program = parse_program(STACK_SOURCE).expect("parses");

        let tree = execute(&program, State::empty()).expect("tree engine runs");
        assert_eq!(tree, State::empty());

        let slot = execute_compiled(&program, State::empty()).expect("slot engine runs");
        assert_eq!(slot, tree);

        let backward = execute_compiled_backward(&program, slot).expect("slot engine reverses");
        assert_eq!(backward, State::empty());
    }

    #[test]
    fn janus_show_records_observations_without_mutating_state() {
        let program = parse_program("show(s, x)").expect("program parses");
        let initial = State::from_bindings([
            ("s".to_owned(), Value::Stack(vec![3, 2, 1])),
            ("x".to_owned(), Value::Int(5)),
        ]);

        let tree = execute_io(&program, IoState::new(initial.clone(), [])).expect("tree runs");
        assert_eq!(tree.store(), &initial);
        assert_eq!(tree.observations(), ["s = <3, 2, 1]", "x = 5"]);

        let slot =
            execute_compiled_io(&program, IoState::new(initial.clone(), [])).expect("slot runs");
        assert_eq!(slot.store(), &initial);
        assert_eq!(slot.observations(), tree.observations());
    }

    #[test]
    fn janus_printf_records_observations_without_mutating_state() {
        let program = parse_program(r#"printf("x=%d, y=%d %%\n", x, y)"#).expect("program parses");
        let initial = State::from_bindings([
            ("x".to_owned(), Value::Int(5)),
            ("y".to_owned(), Value::Int(-2)),
        ]);

        let tree = execute_io(&program, IoState::new(initial.clone(), [])).expect("tree runs");
        assert_eq!(tree.store(), &initial);
        assert_eq!(tree.observations(), ["x=5, y=-2 %\n"]);

        let slot =
            execute_compiled_io(&program, IoState::new(initial.clone(), [])).expect("slot runs");
        assert_eq!(slot.store(), &initial);
        assert_eq!(slot.observations(), tree.observations());
    }

    #[test]
    fn pop_requires_a_zero_target() {
        let program = parse_program("pop(x, s)").expect("program parses");
        let state = State::from_bindings([
            ("x".to_owned(), Value::Int(9)),
            ("s".to_owned(), Value::Stack(vec![1])),
        ]);
        let error = execute(&program, state).expect_err("pop target is non-zero");

        assert!(error.to_string().contains("expected target to be zero"));
    }

    #[test]
    fn fibonacci_pair_transform_runs_forward() {
        let program = parse_program(FIB_SOURCE).expect("parses");
        let state = State::from_bindings([
            ("n".to_owned(), Value::Int(7)),
            ("i".to_owned(), Value::Int(0)),
            ("a".to_owned(), Value::Int(0)),
            ("b".to_owned(), Value::Int(1)),
        ]);
        let state = execute(&program, state).expect("program runs");

        assert_eq!(state.to_string(), "{a = 13, b = 21, i = 7, n = 7}");
    }

    #[test]
    fn fibonacci_pair_transform_runs_backward() {
        let program = parse_program(FIB_SOURCE).expect("parses");
        let state = State::from_bindings([
            ("n".to_owned(), Value::Int(7)),
            ("i".to_owned(), Value::Int(7)),
            ("a".to_owned(), Value::Int(13)),
            ("b".to_owned(), Value::Int(21)),
        ]);
        let state = execute_backward(&program, state).expect("program runs backward");

        assert_eq!(state.to_string(), "{a = 0, b = 1, i = 0, n = 7}");
    }

    #[test]
    fn procedure_with_local_runs_forward_and_backward() {
        let program = parse_program(PROC_SOURCE).expect("parses");
        let initial = State::from_bindings([("n".to_owned(), Value::Int(4))]);

        let forward = execute(&program, initial.clone()).expect("program runs forward");
        assert_eq!(forward.to_string(), "{n = 5}");

        let backward = execute_backward(&program, forward).expect("program runs backward");
        assert_eq!(backward, initial);
    }

    #[test]
    fn slot_compiled_engine_matches_tree_walker_on_examples() {
        let cases = [
            (
                "fib",
                FIB_SOURCE,
                State::from_bindings([
                    ("n".to_owned(), Value::Int(7)),
                    ("i".to_owned(), Value::Int(0)),
                    ("a".to_owned(), Value::Int(0)),
                    ("b".to_owned(), Value::Int(1)),
                ]),
            ),
            (
                "proc",
                PROC_SOURCE,
                State::from_bindings([("n".to_owned(), Value::Int(4))]),
            ),
            (
                "array",
                ARRAY_SOURCE,
                State::from_bindings([
                    (
                        "xs".to_owned(),
                        Value::Array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]),
                    ),
                    ("delta".to_owned(), Value::Int(5)),
                ]),
            ),
            ("negation", NEGATION_SOURCE, State::empty()),
            ("janus_operators", JANUS_OPERATORS_SOURCE, State::empty()),
            (
                "janus_optional_control",
                JANUS_OPTIONAL_CONTROL_SOURCE,
                State::empty(),
            ),
            (
                "janus_procedure_syntax",
                JANUS_PROCEDURE_SYNTAX_SOURCE,
                State::empty(),
            ),
            ("size", SIZE_SOURCE, State::empty()),
            (
                "assert",
                ASSERT_SOURCE,
                State::from_bindings([
                    ("x".to_owned(), Value::Int(0)),
                    ("target".to_owned(), Value::Int(2)),
                ]),
            ),
            ("refinement", REFINEMENT_SOURCE, State::empty()),
        ];

        for (name, source, initial) in cases {
            let program = parse_program(source).expect("example parses");
            let tree = execute(&program, initial.clone()).expect("tree engine runs");
            let slot = execute_compiled(&program, initial).expect("slot engine runs");

            assert_eq!(slot, tree, "{name}");
        }
    }

    #[test]
    fn slot_compiled_engine_runs_backward() {
        let program = parse_program(FIB_SOURCE).expect("parses");
        let final_state = State::from_bindings([
            ("n".to_owned(), Value::Int(7)),
            ("i".to_owned(), Value::Int(7)),
            ("a".to_owned(), Value::Int(13)),
            ("b".to_owned(), Value::Int(21)),
        ]);

        let tree = execute_backward(&program, final_state.clone()).expect("tree engine runs");
        let slot = execute_compiled_backward(&program, final_state).expect("slot engine runs");

        assert_eq!(slot, tree);
    }

    #[test]
    fn procedure_runtime_errors_keep_call_context() {
        let program = parse_program(PROC_RUNTIME_ERROR_SOURCE).expect("parses");
        let error = execute(
            &program,
            State::from_bindings([("n".to_owned(), Value::Int(1))]),
        )
        .expect_err("procedure body fails");

        assert!(error.to_string().contains("division by zero"));
        assert_eq!(error.labels().len(), 1);
        assert_eq!(error.labels()[0].message(), "while calling `boom` here");
    }

    #[test]
    fn procedure_call_can_mutate_array_element_argument() {
        let program = parse_program(
            r#"
proc bump(x) {
  x += 1
}

local xs: array<int> = [41]
  call bump(xs[0])
delocal xs = [42]
"#,
        )
        .expect("program parses");

        let tree = execute(&program, State::empty()).expect("tree engine runs");
        assert_eq!(tree, State::empty());

        let slot = execute_compiled(&program, State::empty()).expect("slot engine runs");
        assert_eq!(slot, tree);
    }

    #[test]
    fn janus_iterate_runs_forward_and_backward() {
        let program = parse_program(
            r#"
iterate int i = 0 to 3
  x += i
end
"#,
        )
        .expect("program parses");
        let initial = State::from_bindings([("x".to_owned(), Value::Int(0))]);

        let tree = execute(&program, initial.clone()).expect("tree engine runs");
        assert_eq!(tree.to_string(), "{x = 6}");

        let slot = execute_compiled(&program, initial.clone()).expect("slot engine runs");
        assert_eq!(slot, tree);

        let backward = execute_backward(&program, tree).expect("tree engine reverses");
        assert_eq!(backward, initial);
    }

    #[test]
    fn janus_iterate_supports_descending_and_empty_ranges() {
        let descending = parse_program(
            r#"
iterate int i = 3 by -1 to 1
  x += i
end
"#,
        )
        .expect("descending iterate parses");
        let empty = parse_program(
            r#"
iterate int i = 0 to -1
  x += 1
end
"#,
        )
        .expect("empty iterate parses");
        let initial = State::from_bindings([("x".to_owned(), Value::Int(0))]);

        let tree = execute(&descending, initial.clone()).expect("tree descending runs");
        assert_eq!(tree.to_string(), "{x = 6}");
        let slot = execute_compiled(&descending, initial.clone()).expect("slot descending runs");
        assert_eq!(slot, tree);

        let tree = execute(&empty, initial.clone()).expect("tree empty runs");
        assert_eq!(tree, initial);
        let slot = execute_compiled(&empty, initial).expect("slot empty runs");
        assert_eq!(slot, tree);
    }

    #[test]
    fn duplicate_resolved_element_arguments_are_rejected() {
        let program = parse_program(
            r#"
proc pair(x, y) {
  x += 1;
  y += 2
}

call pair(xs[i], xs[j])
"#,
        )
        .expect("program parses");
        let state = State::from_bindings([
            ("i".to_owned(), Value::Int(0)),
            ("j".to_owned(), Value::Int(0)),
            ("xs".to_owned(), Value::Array(vec![Value::Int(0)])),
        ]);

        let error = execute(&program, state.clone()).expect_err("tree rejects duplicate cell");
        assert!(error.to_string().contains("duplicate argument `xs[0]`"));

        let error = execute_compiled(&program, state).expect_err("slot rejects duplicate cell");
        assert!(error.to_string().contains("duplicate argument `xs[0]`"));
    }

    #[test]
    fn uncall_runs_a_procedure_body_backward() {
        let program = parse_program(
            r#"
proc bump(x) {
  local t = 0
    t += x;
    x += 1
  delocal t = x - 1
}

uncall bump(n)
"#,
        )
        .expect("parses");
        let state = State::from_bindings([("n".to_owned(), Value::Int(5))]);
        let state = execute(&program, state).expect("program runs");

        assert_eq!(state.to_string(), "{n = 4}");
    }

    #[test]
    fn refinement_example_runs_forward_and_backward() {
        let program = parse_program(REFINEMENT_SOURCE).expect("parses");
        let forward = execute(&program, State::empty()).expect("program runs forward");
        assert_eq!(forward, State::empty());

        let backward = execute_backward(&program, forward).expect("program runs backward");
        assert_eq!(backward, State::empty());
    }

    #[test]
    fn refinement_violation_is_a_runtime_error() {
        let program = parse_program(REFINEMENT_VIOLATION_SOURCE).expect("parses");
        let error = execute(&program, State::empty()).expect_err("program violates refinement");

        assert!(error.to_string().contains("refinement for `n`"));
    }

    #[test]
    fn timeline_records_forward_states() {
        let program = parse_program("x += 1; x <=> y").expect("parses");
        let initial = State::from_bindings([
            ("x".to_owned(), Value::Int(1)),
            ("y".to_owned(), Value::Int(9)),
        ]);
        let timeline = build_timeline(&program, initial).expect("timeline builds");

        assert_eq!(timeline.len(), 3);
        assert_eq!(timeline.frames()[0].state.to_string(), "{x = 1, y = 9}");
        assert_eq!(timeline.frames()[1].state.to_string(), "{x = 2, y = 9}");
        assert_eq!(timeline.frames()[2].state.to_string(), "{x = 9, y = 2}");
    }

    #[test]
    fn timeline_final_state_matches_execute() {
        let program = parse_program(FIB_SOURCE).expect("parses");
        let initial = State::from_bindings([
            ("n".to_owned(), Value::Int(7)),
            ("i".to_owned(), Value::Int(0)),
            ("a".to_owned(), Value::Int(0)),
            ("b".to_owned(), Value::Int(1)),
        ]);
        let final_state = execute(&program, initial.clone()).expect("program runs");
        let timeline = build_timeline(&program, initial).expect("timeline builds");

        assert_eq!(timeline.final_state(), Some(&final_state));
    }

    #[test]
    fn timeline_supports_tape_io() {
        let program =
            parse_program("global x; proc main() { write x; read x; write x }").expect("parses");
        let initial = IoState::new(State::empty(), [Value::Int(7)]);
        let timeline = build_timeline_io(&program, initial).expect("timeline builds");

        assert!(
            timeline
                .frames()
                .iter()
                .any(|frame| frame.label == "write x = 0")
        );
        assert!(
            timeline
                .frames()
                .iter()
                .any(|frame| frame.label == "read x <- 7" && frame.state.to_string() == "{x = 7}")
        );
        assert_eq!(
            timeline.final_state().map(ToString::to_string),
            Some("{x = 7}".to_owned())
        );
    }

    #[test]
    fn timeline_traces_procedure_body_in_local_context() {
        let program = parse_program(PROC_SOURCE).expect("parses");
        let timeline = build_timeline(
            &program,
            State::from_bindings([("n".to_owned(), Value::Int(4))]),
        )
        .expect("timeline builds");
        let frames = timeline.frames();

        assert_eq!(
            frames.first().expect("start frame").state.to_string(),
            "{n = 4}"
        );
        assert!(
            frames
                .iter()
                .any(|frame| frame.label == "call bump enter"
                    && frame.state.to_string() == "{x = 4}")
        );
        assert!(
            frames
                .iter()
                .any(|frame| frame.label == "x +=" && frame.state.to_string() == "{t = 4, x = 5}")
        );
        assert!(
            frames.iter().any(
                |frame| frame.label == "call bump exit" && frame.state.to_string() == "{n = 5}"
            )
        );
        assert_eq!(
            timeline.final_state().map(ToString::to_string),
            Some("{n = 5}".to_owned())
        );
    }

    proptest! {
        #[test]
        fn straight_line_programs_round_trip(program in straight_line_program(), state in small_int_state()) {
            let forward = execute(&program, state.clone()).expect("generated program runs forward");
            let backward = execute_backward(&program, forward).expect("generated program runs backward");

            prop_assert_eq!(backward, state);
        }

        #[test]
        fn slot_compiled_matches_tree_for_generated_straight_line(program in straight_line_program(), state in small_int_state()) {
            let tree = execute(&program, state.clone()).expect("tree engine runs generated program");
            let slot = execute_compiled(&program, state).expect("slot engine runs generated program");

            prop_assert_eq!(slot, tree);
        }

        #[test]
        fn array_straight_line_programs_round_trip(program in array_straight_line_program(), state in small_array_state()) {
            reverie_core::check_program(&program).expect("generated array program checks");

            let forward = execute(&program, state.clone()).expect("generated array program runs forward");
            let backward = execute_backward(&program, forward).expect("generated array program runs backward");

            prop_assert_eq!(backward, state);
        }

        #[test]
        fn slot_compiled_matches_tree_for_generated_arrays(program in array_straight_line_program(), state in small_array_state()) {
            reverie_core::check_program(&program).expect("generated array program checks");

            let tree = execute(&program, state.clone()).expect("tree engine runs generated array program");
            let slot = execute_compiled(&program, state).expect("slot engine runs generated array program");

            prop_assert_eq!(slot, tree);
        }

        #[test]
        fn bool_array_programs_round_trip(program in bool_array_straight_line_program(), state in small_bool_array_state()) {
            reverie_core::check_program(&program).expect("generated bool array program checks");

            let forward = execute(&program, state.clone()).expect("generated bool array program runs forward");
            let backward = execute_backward(&program, forward).expect("generated bool array program runs backward");

            prop_assert_eq!(backward, state);
        }

        #[test]
        fn slot_compiled_matches_tree_for_generated_bool_arrays(program in bool_array_straight_line_program(), state in small_bool_array_state()) {
            reverie_core::check_program(&program).expect("generated bool array program checks");

            let tree = execute(&program, state.clone()).expect("tree engine runs generated bool array program");
            let slot = execute_compiled(&program, state).expect("slot engine runs generated bool array program");

            prop_assert_eq!(slot, tree);
        }

        #[test]
        fn simple_procedure_calls_round_trip(initial in any::<i64>(), amount in 0_i64..=64_i64) {
            let source = format!(
                r#"
proc adjust(x) {{
  x += {amount}
}}

call adjust(n)
"#
            );
            let program = parse_program(&source).expect("program parses");
            let state = State::from_bindings([("n".to_owned(), Value::Int(initial))]);

            let forward = execute(&program, state.clone()).expect("generated procedure runs forward");
            let backward = execute_backward(&program, forward).expect("generated procedure runs backward");

            prop_assert_eq!(backward, state);
        }

        #[test]
        fn uncall_matches_inverted_body(initial in any::<i64>(), amount in any::<i64>(), op in 0_u8..=2_u8) {
            let amount = source_int_literal(amount);
            let (forward_op, inverse_op) = match op {
                0 => ("+=", "-="),
                1 => ("-=", "+="),
                _ => ("^=", "^="),
            };
            let uncall_source = format!(
                r#"
proc adjust(x) {{
  x {forward_op} {amount}
}}

uncall adjust(n)
"#
            );
            let inverted_body_source = format!("n {inverse_op} {amount}");
            let uncall_program = parse_program(&uncall_source).expect("uncall program parses");
            let inverted_body = parse_program(&inverted_body_source).expect("inverted body parses");
            let state = State::from_bindings([("n".to_owned(), Value::Int(initial))]);

            let through_uncall = execute(&uncall_program, state.clone()).expect("uncall runs");
            let through_inverted_body = execute(&inverted_body, state).expect("inverted body runs");

            prop_assert_eq!(through_uncall, through_inverted_body);
        }

        #[test]
        fn generated_conditionals_round_trip(flag in any::<bool>(), x in any::<i64>(), y in any::<i64>(), amount in -64_i64..=64_i64) {
            let amount = source_int_literal(amount);
            let source = format!(
                r#"
if flag then
  x += {amount}
else
  y -= {amount}
fi flag
"#
            );
            let program = parse_program(&source).expect("generated conditional parses");
            let state = State::from_bindings([
                ("flag".to_owned(), Value::Bool(flag)),
                ("x".to_owned(), Value::Int(x)),
                ("y".to_owned(), Value::Int(y)),
            ]);

            let forward = execute(&program, state.clone()).expect("generated conditional runs forward");
            let backward = execute_backward(&program, forward).expect("generated conditional runs backward");

            prop_assert_eq!(backward, state);
        }

        #[test]
        fn generated_loops_round_trip(n in 1_i64..=64_i64, acc in any::<i64>(), delta in any::<i64>()) {
            let source = r#"
from i == 0 do
  acc += delta;
  i += 1
loop
  skip
until i == n
"#;
            let program = parse_program(source).expect("generated loop parses");
            let state = State::from_bindings([
                ("n".to_owned(), Value::Int(n)),
                ("i".to_owned(), Value::Int(0)),
                ("acc".to_owned(), Value::Int(acc)),
                ("delta".to_owned(), Value::Int(delta)),
            ]);

            let forward = execute(&program, state.clone()).expect("generated loop runs forward");
            let backward = execute_backward(&program, forward).expect("generated loop runs backward");

            prop_assert_eq!(backward, state);
        }

        #[test]
        fn generated_local_array_programs_round_trip(xs in prop::array::uniform3(any::<i64>()), delta in any::<i64>()) {
            let final_xs = [xs[2], xs[1].wrapping_add(delta), xs[0]];
            let source = format!(
                r#"
local xs: array<int> = [{}, {}, {}]
  local delta: int = {}
    xs[1] += delta;
    xs[0] <=> xs[2]
  delocal delta = {}
delocal xs = [{}, {}, {}]
"#,
                source_int_literal(xs[0]),
                source_int_literal(xs[1]),
                source_int_literal(xs[2]),
                source_int_literal(delta),
                source_int_literal(delta),
                source_int_literal(final_xs[0]),
                source_int_literal(final_xs[1]),
                source_int_literal(final_xs[2]),
            );
            let program = parse_program(&source).expect("generated local array program parses");
            reverie_core::check_program(&program).expect("generated local array program checks");

            let forward = execute(&program, State::empty()).expect("generated local array program runs forward");
            let backward = execute_backward(&program, forward.clone()).expect("generated local array program runs backward");

            prop_assert_eq!(forward, State::empty());
            prop_assert_eq!(backward, State::empty());
        }
    }

    fn small_int_state() -> impl Strategy<Value = State> {
        (any::<i64>(), any::<i64>(), any::<i64>()).prop_map(|(x, y, z)| {
            State::from_bindings([
                ("x".to_owned(), Value::Int(x)),
                ("y".to_owned(), Value::Int(y)),
                ("z".to_owned(), Value::Int(z)),
            ])
        })
    }

    fn small_array_state() -> impl Strategy<Value = State> {
        (
            any::<i64>(),
            any::<i64>(),
            any::<i64>(),
            prop::array::uniform3(any::<i64>()),
        )
            .prop_map(|(x, y, z, xs)| {
                State::from_bindings([
                    ("x".to_owned(), Value::Int(x)),
                    ("y".to_owned(), Value::Int(y)),
                    ("z".to_owned(), Value::Int(z)),
                    (
                        "xs".to_owned(),
                        Value::Array(xs.into_iter().map(Value::Int).collect()),
                    ),
                ])
            })
    }

    fn small_bool_array_state() -> impl Strategy<Value = State> {
        (any::<bool>(), prop::array::uniform3(any::<bool>())).prop_map(|(flag, flags)| {
            State::from_bindings([
                ("flag".to_owned(), Value::Bool(flag)),
                (
                    "flags".to_owned(),
                    Value::Array(flags.into_iter().map(Value::Bool).collect()),
                ),
            ])
        })
    }

    fn straight_line_program() -> impl Strategy<Value = Program> {
        prop::collection::vec(primitive_stmt(), 1..32)
            .prop_map(|statements| Program::new(stmt_sequence(statements)))
    }

    fn array_straight_line_program() -> impl Strategy<Value = Program> {
        prop::collection::vec(array_primitive_stmt(), 1..32)
            .prop_map(|statements| Program::new(stmt_sequence(statements)))
    }

    fn bool_array_straight_line_program() -> impl Strategy<Value = Program> {
        prop::collection::vec(bool_array_primitive_stmt(), 1..32)
            .prop_map(|statements| Program::new(stmt_sequence(statements)))
    }

    fn stmt_sequence(statements: Vec<SpannedStmt>) -> SpannedStmt {
        if statements.len() == 1 {
            statements
                .into_iter()
                .next()
                .expect("at least one generated statement")
        } else {
            let span = statements
                .first()
                .map(|statement| statement.span.start)
                .unwrap_or(0)
                ..statements
                    .last()
                    .map(|statement| statement.span.end)
                    .unwrap_or(0);
            Spanned::new(Stmt::Seq(statements), span)
        }
    }

    fn primitive_stmt() -> impl Strategy<Value = SpannedStmt> {
        prop_oneof![
            Just(Spanned::new(Stmt::Skip, 0..0)),
            update_stmt(),
            swap_stmt(),
        ]
    }

    fn array_primitive_stmt() -> impl Strategy<Value = SpannedStmt> {
        prop_oneof![
            Just(Spanned::new(Stmt::Skip, 0..0)),
            array_update_stmt(),
            array_swap_stmt(),
        ]
    }

    fn bool_array_primitive_stmt() -> impl Strategy<Value = SpannedStmt> {
        prop_oneof![
            Just(Spanned::new(Stmt::Skip, 0..0)),
            bool_array_update_stmt(),
            bool_array_swap_stmt(),
        ]
    }

    fn update_stmt() -> impl Strategy<Value = SpannedStmt> {
        let targets = prop_oneof![
            Just(Place::new("x".to_owned(), None)),
            Just(Place::new("y".to_owned(), None)),
            Just(Place::new("z".to_owned(), None))
        ];
        let ops = prop_oneof![
            Just(UpdateOp::Add),
            Just(UpdateOp::Sub),
            Just(UpdateOp::Xor),
        ];

        (targets, ops, -64_i64..=64_i64).prop_map(|(target, op, amount)| {
            Spanned::new(
                Stmt::Update {
                    target,
                    op,
                    expr: Spanned::new(
                        Expr::Int {
                            value: amount,
                            unit: None,
                        },
                        0..0,
                    ),
                },
                0..0,
            )
        })
    }

    fn swap_stmt() -> impl Strategy<Value = SpannedStmt> {
        prop_oneof![
            Just((
                Place::new("x".to_owned(), None),
                Place::new("y".to_owned(), None)
            )),
            Just((
                Place::new("x".to_owned(), None),
                Place::new("z".to_owned(), None)
            )),
            Just((
                Place::new("y".to_owned(), None),
                Place::new("z".to_owned(), None)
            )),
        ]
        .prop_map(|(left, right)| Spanned::new(Stmt::Swap { left, right }, 0..0))
    }

    fn array_update_stmt() -> impl Strategy<Value = SpannedStmt> {
        prop_oneof![
            (
                Just(Place::new("x".to_owned(), None)),
                update_op(),
                rhs_expr_excluding("x")
            )
                .prop_map(update_from_parts),
            (
                Just(Place::new("y".to_owned(), None)),
                update_op(),
                rhs_expr_excluding("y")
            )
                .prop_map(update_from_parts),
            (
                Just(Place::new("z".to_owned(), None)),
                update_op(),
                rhs_expr_excluding("z")
            )
                .prop_map(update_from_parts),
            (array_place(), update_op(), rhs_expr_excluding("xs")).prop_map(update_from_parts),
        ]
    }

    fn update_op() -> impl Strategy<Value = UpdateOp> {
        prop_oneof![
            Just(UpdateOp::Add),
            Just(UpdateOp::Sub),
            Just(UpdateOp::Xor),
        ]
    }

    fn rhs_expr_excluding(target: &'static str) -> BoxedStrategy<SpannedExpr> {
        let literal = (-64_i64..=64_i64).prop_map(int_expr);
        match target {
            "x" => prop_oneof![
                literal.boxed(),
                Just(var_expr("y")).boxed(),
                Just(var_expr("z")).boxed()
            ]
            .boxed(),
            "y" => prop_oneof![
                literal.boxed(),
                Just(var_expr("x")).boxed(),
                Just(var_expr("z")).boxed()
            ]
            .boxed(),
            "z" => prop_oneof![
                literal.boxed(),
                Just(var_expr("x")).boxed(),
                Just(var_expr("y")).boxed()
            ]
            .boxed(),
            "xs" => prop_oneof![
                literal.boxed(),
                Just(var_expr("x")).boxed(),
                Just(var_expr("y")).boxed(),
                Just(var_expr("z")).boxed()
            ]
            .boxed(),
            _ => literal.boxed(),
        }
    }

    fn update_from_parts((target, op, expr): (Place, UpdateOp, SpannedExpr)) -> SpannedStmt {
        Spanned::new(Stmt::Update { target, op, expr }, 0..0)
    }

    fn array_swap_stmt() -> impl Strategy<Value = SpannedStmt> {
        (int_place(), int_place())
            .prop_map(|(left, right)| Spanned::new(Stmt::Swap { left, right }, 0..0))
    }

    fn bool_array_update_stmt() -> impl Strategy<Value = SpannedStmt> {
        (bool_place(), any::<bool>()).prop_map(|(target, value)| {
            Spanned::new(
                Stmt::Update {
                    target,
                    op: UpdateOp::Xor,
                    expr: bool_expr(value),
                },
                0..0,
            )
        })
    }

    fn bool_array_swap_stmt() -> impl Strategy<Value = SpannedStmt> {
        prop_oneof![
            Just((bool_scalar_place(), bool_array_place(0))),
            Just((bool_scalar_place(), bool_array_place(1))),
            Just((bool_scalar_place(), bool_array_place(2))),
            Just((bool_array_place(0), bool_array_place(1))),
            Just((bool_array_place(0), bool_array_place(2))),
            Just((bool_array_place(1), bool_array_place(2))),
        ]
        .prop_map(|(left, right)| Spanned::new(Stmt::Swap { left, right }, 0..0))
    }

    fn int_place() -> impl Strategy<Value = Place> {
        prop_oneof![
            Just(Place::new("x".to_owned(), None)),
            Just(Place::new("y".to_owned(), None)),
            Just(Place::new("z".to_owned(), None)),
            array_place(),
        ]
    }

    fn array_place() -> impl Strategy<Value = Place> {
        (0_i64..=2_i64).prop_map(|index| Place::new("xs".to_owned(), Some(int_expr(index))))
    }

    fn bool_place() -> impl Strategy<Value = Place> {
        prop_oneof![
            Just(bool_scalar_place()),
            Just(bool_array_place(0)),
            Just(bool_array_place(1)),
            Just(bool_array_place(2)),
        ]
    }

    fn bool_scalar_place() -> Place {
        Place::new("flag".to_owned(), None)
    }

    fn bool_array_place(index: i64) -> Place {
        Place::new("flags".to_owned(), Some(int_expr(index)))
    }

    fn int_expr(value: i64) -> SpannedExpr {
        Spanned::new(Expr::Int { value, unit: None }, 0..0)
    }

    fn bool_expr(value: bool) -> SpannedExpr {
        Spanned::new(Expr::Bool(value), 0..0)
    }

    fn source_int_literal(value: i64) -> String {
        match value {
            i64::MIN => "(0 - 9223372036854775807 - 1)".to_owned(),
            value if value < 0 => format!("(0 - {})", -(value as i128)),
            value => value.to_string(),
        }
    }

    fn var_expr(name: &str) -> SpannedExpr {
        Spanned::new(Expr::Var(name.to_owned()), 0..0)
    }
}
