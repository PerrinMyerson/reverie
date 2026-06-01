use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ops::Range;

use reverie_syntax::{
    BinaryOp, Expr, GlobalDecl, Place, Proc, Program, SpannedExpr, SpannedStmt, Stmt, TypeExpr,
    UnaryOp, UpdateOp,
};

use crate::{
    IoState, RuntimeError, State, Value, format_printf_observation, format_show_observation,
};

#[derive(Debug, Clone)]
pub struct CompiledProgram {
    globals: Vec<CGlobal>,
    top: CompiledScope,
    procedures: Vec<CompiledProcedure>,
}

impl CompiledProgram {
    pub fn for_state(program: &Program, initial_state: &State) -> Result<Self, RuntimeError> {
        let mut top_layout = SlotLayout::default();
        for global in &program.globals {
            top_layout.ensure(&global.name);
        }
        for name in initial_state.store().keys() {
            top_layout.ensure(name);
        }
        collect_stmt_names(&program.body, &mut top_layout);

        let mut procedure_layouts = BTreeMap::new();
        for procedure in &program.procedures {
            if procedure_layouts.contains_key(&procedure.name) {
                return Err(RuntimeError::new(format!(
                    "duplicate procedure `{}`",
                    procedure.name
                )));
            }

            let mut layout = SlotLayout::default();
            for global in &program.globals {
                layout.ensure(&global.name);
            }
            for param in &procedure.params {
                layout.ensure(&param.name);
            }
            collect_stmt_names(&procedure.body, &mut layout);
            procedure_layouts.insert(procedure.name.clone(), layout);
        }

        let mut procedures = Vec::new();
        let mut procedure_indices = HashMap::new();
        for procedure in &program.procedures {
            let layout = procedure_layouts
                .get(&procedure.name)
                .expect("procedure layout collected")
                .clone();
            let procedure = compile_procedure(procedure, &program.globals, layout)?;
            let procedure_index = procedures.len();
            if procedure_indices
                .insert(procedure.name.clone(), procedure_index)
                .is_some()
            {
                return Err(RuntimeError::new("duplicate compiled procedure"));
            }
            procedures.push(procedure);
        }
        propagate_procedure_global_effects(&mut procedures, &program.globals);
        propagate_procedure_param_effects(&mut procedures, &program.globals);
        refresh_procedure_runtime_indices(&mut procedures);
        let direct_frame_signatures = direct_frame_signatures(&procedures);
        refresh_procedure_call_sites(
            &mut procedures,
            &procedure_indices,
            &direct_frame_signatures,
        )?;

        let mut top_body = compile_stmt(&program.body, &top_layout)?;
        refresh_call_metadata(&mut top_body, &procedure_indices, &direct_frame_signatures)?;

        Ok(Self {
            globals: program
                .globals
                .iter()
                .map(|global| {
                    Ok(CGlobal {
                        name: global.name.clone(),
                        slot: top_layout.resolve(&global.name)?,
                        referenced: true,
                        mutable: true,
                        dims: global.dims.clone(),
                        ty: global.ty.as_ref().map(|ty| ty.node.clone()),
                        init: global
                            .init
                            .as_ref()
                            .map(|expr| compile_expr(expr, &top_layout))
                            .transpose()?,
                    })
                })
                .collect::<Result<Vec<_>, RuntimeError>>()?,
            top: CompiledScope {
                layout: top_layout.clone(),
                body: top_body,
            },
            procedures,
        })
    }

    pub fn execute(&self, state: State) -> Result<State, RuntimeError> {
        self.execute_io(IoState::new(state, []))
            .map(IoState::into_store)
    }

    pub fn execute_io(&self, state: IoState) -> Result<IoState, RuntimeError> {
        let (store, mut input, mut output, mut observations) = state.into_parts();
        let mut state = SlotState::from_external(&self.top.layout, store)?;
        initialize_globals(&self.globals, &mut state)?;
        let mut tapes = Tapes {
            input: &mut input,
            output: &mut output,
            observations: &mut observations,
        };
        self.execute_stmt(&self.top.body, &mut state, &mut tapes)?;
        Ok(IoState::from_parts(
            state.to_external(&self.top.layout),
            input,
            output,
            observations,
        ))
    }

    fn execute_stmt(
        &self,
        statement: &CStmt,
        state: &mut SlotState,
        tapes: &mut Tapes<'_>,
    ) -> Result<(), RuntimeError> {
        self.execute_stmt_inner(statement, state, tapes)
            .map_err(|error| error.with_span_if_missing(statement.span.clone()))
    }

    fn execute_stmt_inner(
        &self,
        statement: &CStmt,
        state: &mut SlotState,
        tapes: &mut Tapes<'_>,
    ) -> Result<(), RuntimeError> {
        match &statement.kind {
            CStmtKind::Skip => Ok(()),
            CStmtKind::Assert { condition } => {
                assert_condition(condition, state, true, "assertion")
            }
            CStmtKind::Seq(statements) => {
                for statement in statements {
                    self.execute_stmt(statement, state, tapes)?;
                }
                Ok(())
            }
            CStmtKind::Update {
                target,
                op,
                expr,
                alias_check,
            } => {
                if *alias_check {
                    ensure_update_rhs_does_not_read_target(target, expr, state)?;
                }
                match op {
                    UpdateOp::Add | UpdateOp::Sub => {
                        let rhs = eval_int(expr, state)?;
                        update_place_int(target, state, *op, rhs)
                    }
                    UpdateOp::Xor => {
                        let rhs = eval(expr, state)?;
                        update_place_xor(target, state, rhs)
                    }
                }
            }
            CStmtKind::Swap { left, right } => swap_places(left, right, state),
            CStmtKind::Push { source, stack } => push_stack(source, stack, state),
            CStmtKind::Pop { target, stack } => pop_stack(target, stack, state),
            CStmtKind::If {
                entry,
                then_branch,
                else_branch,
                exit,
            } => {
                if eval_bool(entry, state)? {
                    self.execute_stmt(then_branch, state, tapes)?;
                    assert_condition(exit, state, true, "if exit assertion")
                } else {
                    self.execute_stmt(else_branch, state, tapes)?;
                    assert_condition(exit, state, false, "if exit assertion")
                }
            }
            CStmtKind::Loop {
                entry,
                body,
                step,
                exit,
            } => {
                assert_condition(entry, state, true, "loop entry assertion")?;

                loop {
                    self.execute_stmt(body, state, tapes)?;
                    if eval_bool(exit, state)? {
                        return Ok(());
                    }

                    self.execute_stmt(step, state, tapes)?;
                    assert_condition(entry, state, false, "loop re-entry assertion")?;
                }
            }
            CStmtKind::Iterate {
                name,
                slot,
                start,
                step,
                end,
                body,
            } => self.execute_iterate(
                CIterateRuntime {
                    name,
                    slot: *slot,
                    start,
                    step,
                    end,
                    body,
                },
                statement.span.clone(),
                state,
                tapes,
            ),
            CStmtKind::Call {
                name,
                target,
                args,
                runtime_arg_check,
                direct_frame,
            } => self.execute_procedure(
                CProcedureRuntime {
                    name,
                    target: *target,
                    args,
                    runtime_arg_check: *runtime_arg_check,
                    direct_frame: *direct_frame,
                    direction: Direction::Call,
                },
                statement.span.clone(),
                state,
                tapes,
            ),
            CStmtKind::Uncall {
                name,
                target,
                args,
                runtime_arg_check,
                direct_frame,
            } => self.execute_procedure(
                CProcedureRuntime {
                    name,
                    target: *target,
                    args,
                    runtime_arg_check: *runtime_arg_check,
                    direct_frame: *direct_frame,
                    direction: Direction::Uncall,
                },
                statement.span.clone(),
                state,
                tapes,
            ),
            CStmtKind::Read { target } => {
                let next = tapes.input.pop_front().ok_or_else(|| {
                    RuntimeError::at("read expected an input tape value", statement.span.clone())
                })?;
                let previous = eval_place(target, state)?;
                assign_place(target, state, next)?;
                tapes.output.push(previous);
                Ok(())
            }
            CStmtKind::Unread { target } => {
                let previous = tapes.output.pop().ok_or_else(|| {
                    RuntimeError::at(
                        "unread expected an output tape value",
                        statement.span.clone(),
                    )
                })?;
                let current = eval_place(target, state)?;
                assign_place(target, state, previous)?;
                tapes.input.push_front(current);
                Ok(())
            }
            CStmtKind::Write { source } => {
                tapes.output.push(eval_place(source, state)?);
                Ok(())
            }
            CStmtKind::Unwrite { source } => {
                let expected = eval_place(source, state)?;
                let actual = tapes.output.pop().ok_or_else(|| {
                    RuntimeError::at(
                        "unwrite expected an output tape value",
                        statement.span.clone(),
                    )
                })?;
                if actual == expected {
                    Ok(())
                } else {
                    Err(RuntimeError::at(
                        format!("unwrite expected {expected}, found {actual}"),
                        statement.span.clone(),
                    ))
                }
            }
            CStmtKind::Show { targets } => {
                for (target, slot) in targets {
                    let value = state.value(*slot, target)?;
                    tapes
                        .observations
                        .push(format_show_observation(target, value));
                }
                Ok(())
            }
            CStmtKind::Printf { format, args } => {
                let values = args
                    .iter()
                    .map(|arg| eval_int(arg, state))
                    .collect::<Result<Vec<_>, _>>()?;
                tapes.observations.push(format_printf_observation(
                    format,
                    &values,
                    statement.span.clone(),
                )?);
                Ok(())
            }
            CStmtKind::Declare {
                name,
                slot,
                ty,
                dims,
                init,
            } => declare_value(name, *slot, ty, dims, init.as_ref(), state),
            CStmtKind::Local {
                name,
                slot,
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
            } => {
                if name != delocal_name {
                    return Err(RuntimeError::new(format!(
                        "local `{name}` must be destroyed by matching `delocal {name}`"
                    )));
                }

                if state.is_initialized(*slot) {
                    return Err(RuntimeError::new(format!(
                        "local `{name}` would shadow an existing variable"
                    )));
                }

                let init_value = eval(init, state)?;
                validate_local_value(name, dims, ty.as_ref(), &init_value)?;
                state.set(*slot, init_value)?;
                assert_refinement(name, refinement.as_ref(), state)?;
                self.execute_stmt(body, state, tapes)?;
                assert_refinement(name, refinement.as_ref(), state)?;

                let actual = state.value(*slot, name)?.clone();
                let expected = eval(delocal, state)?;
                validate_local_value(name, dims, ty.as_ref(), &actual)?;
                validate_local_value(name, dims, ty.as_ref(), &expected)?;
                validate_local_value(name, delocal_dims, delocal_ty.as_ref(), &actual)?;
                validate_local_value(name, delocal_dims, delocal_ty.as_ref(), &expected)?;
                if actual != expected {
                    return Err(RuntimeError::at(
                        format!("delocal `{name}` expected {expected}, found {actual}"),
                        delocal.span.clone(),
                    ));
                }
                assert_refinement(name, delocal_refinement.as_ref(), state)?;

                state.clear(*slot)?;
                Ok(())
            }
        }
    }

    fn execute_procedure(
        &self,
        call: CProcedureRuntime<'_>,
        call_span: Range<usize>,
        caller_state: &mut SlotState,
        tapes: &mut Tapes<'_>,
    ) -> Result<(), RuntimeError> {
        let name = call.name;
        let Some(target) = call.target else {
            return Err(RuntimeError::new(format!("unknown procedure `{name}`")));
        };
        let procedure = self
            .procedures
            .get(target)
            .ok_or_else(|| RuntimeError::new(format!("unknown procedure `{name}`")))?;

        if procedure.params.len() != call.args.len() {
            return Err(RuntimeError::at(
                format!(
                    "procedure `{name}` expects {} argument(s), found {}",
                    procedure.params.len(),
                    call.args.len()
                ),
                procedure.span.clone(),
            ));
        }

        let frozen_args = if call.runtime_arg_check {
            Some(freeze_call_args(call.args, caller_state)?)
        } else {
            None
        };
        let args = frozen_args
            .as_ref()
            .map_or(call.args, FrozenCallArgs::as_slice);
        if call.runtime_arg_check {
            ensure_unique_args(name, args)?;
        }

        if call.direct_frame {
            let result = (|| {
                if procedure.has_param_refinements {
                    for param in &procedure.params {
                        assert_refinement(&param.name, param.refinement.as_ref(), caller_state)?;
                    }
                }

                match call.direction {
                    Direction::Call => self.execute_stmt(&procedure.body, caller_state, tapes)?,
                    Direction::Uncall => {
                        self.execute_stmt(&procedure.inverse_body, caller_state, tapes)?
                    }
                }

                if procedure.has_param_refinements {
                    for param in &procedure.params {
                        assert_refinement(&param.name, param.refinement.as_ref(), caller_state)?;
                    }
                }

                Ok(())
            })();

            return result.map_err(|error: RuntimeError| {
                error.with_context_label(
                    call_span,
                    format!("while {} `{name}` here", call.direction.gerund()),
                )
            });
        }

        let mut copied_global_indices = Vec::new();
        let result: Result<SlotState, RuntimeError> = (|| {
            let mut local_state = SlotState::empty(procedure.layout.len());
            for global_index in &procedure.active_global_indices {
                let global = &procedure.globals[*global_index];
                if args.iter().any(|arg| arg.slot == global.slot) {
                    continue;
                }

                if caller_state.is_initialized(global.slot) {
                    let value = caller_state.value(global.slot, &global.name)?.clone();
                    if global.mutable {
                        copied_global_indices.push(*global_index);
                    }
                    local_state.set(global.slot, value)?;
                }
            }

            for (param, arg) in procedure.params.iter().zip(args) {
                let value = eval_place(arg, caller_state)?;
                local_state.set(param.slot, value)?;
            }

            if procedure.has_param_refinements {
                for param in &procedure.params {
                    assert_refinement(&param.name, param.refinement.as_ref(), &local_state)?;
                }
            }

            match call.direction {
                Direction::Call => self.execute_stmt(&procedure.body, &mut local_state, tapes)?,
                Direction::Uncall => {
                    self.execute_stmt(&procedure.inverse_body, &mut local_state, tapes)?
                }
            }

            if procedure.has_param_refinements {
                for param in &procedure.params {
                    assert_refinement(&param.name, param.refinement.as_ref(), &local_state)?;
                }
            }

            Ok(local_state)
        })();

        let local_state = result.map_err(|error| {
            error.with_context_label(
                call_span,
                format!("while {} `{name}` here", call.direction.gerund()),
            )
        })?;

        for index in &procedure.mutable_param_indices {
            let param = &procedure.params[*index];
            let arg = &args[*index];
            let value = local_state.value(param.slot, &param.name)?.clone();
            assign_place(arg, caller_state, value)?;
        }

        for global_index in copied_global_indices {
            let global = &procedure.globals[global_index];
            let value = local_state.value(global.slot, &global.name)?.clone();
            caller_state.set(global.slot, value)?;
        }

        Ok(())
    }

    fn execute_iterate(
        &self,
        iterate: CIterateRuntime<'_>,
        span: Range<usize>,
        state: &mut SlotState,
        tapes: &mut Tapes<'_>,
    ) -> Result<(), RuntimeError> {
        let bounds = eval_iterate_bounds(iterate, state, span.clone())?;
        if state.is_initialized(iterate.slot) {
            return Err(RuntimeError::new(format!(
                "iterate variable `{}` would shadow an existing variable",
                iterate.name
            )));
        }

        state.set(iterate.slot, Value::Int(bounds.start))?;
        let mut current = bounds.start;
        while iterate_contains(current, bounds.end, bounds.step) {
            state.set(iterate.slot, Value::Int(current))?;
            self.execute_stmt(iterate.body, state, tapes)?;
            assert_iterate_variable(iterate.name, iterate.slot, current, state, span.clone())?;
            let Some(next) = next_iterate_value(current, bounds.end, bounds.step, span.clone())?
            else {
                break;
            };
            current = next;
        }
        state.clear(iterate.slot)
    }
}

pub fn execute_compiled(program: &Program, state: State) -> Result<State, RuntimeError> {
    let compiled = CompiledProgram::for_state(program, &state)?;
    compiled.execute(state)
}

pub fn execute_compiled_io(program: &Program, state: IoState) -> Result<IoState, RuntimeError> {
    let compiled = CompiledProgram::for_state(program, state.store())?;
    compiled.execute_io(state)
}

pub fn execute_compiled_backward(program: &Program, state: State) -> Result<State, RuntimeError> {
    let inverse = reverie_core::invert_program(program);
    execute_compiled(&inverse, state)
}

pub fn execute_compiled_io_backward(
    program: &Program,
    state: IoState,
) -> Result<IoState, RuntimeError> {
    let inverse = reverie_core::invert_program(program);
    execute_compiled_io(&inverse, state)
}

#[derive(Debug, Clone)]
struct CompiledScope {
    layout: SlotLayout,
    body: CStmt,
}

#[derive(Debug, Clone)]
struct CompiledProcedure {
    name: String,
    params: Vec<CParam>,
    mutable_param_indices: Vec<usize>,
    active_global_indices: Vec<usize>,
    direct_frame_eligible: bool,
    has_param_refinements: bool,
    globals: Vec<CGlobal>,
    body: CStmt,
    inverse_body: CStmt,
    call_sites: Vec<CCallSite>,
    call_targets: Vec<String>,
    layout: SlotLayout,
    span: Range<usize>,
}

impl CompiledProcedure {
    fn direct_frame_signature(&self) -> DirectFrameSignature {
        DirectFrameSignature {
            eligible: self.direct_frame_eligible,
            param_slots: self.params.iter().map(|param| param.slot).collect(),
        }
    }
}

#[derive(Debug, Clone)]
struct DirectFrameSignature {
    eligible: bool,
    param_slots: Vec<usize>,
}

#[derive(Debug, Clone)]
struct CParam {
    name: String,
    slot: usize,
    mutable: bool,
    refinement: Option<CExpr>,
}

#[derive(Debug, Clone)]
struct CGlobal {
    name: String,
    slot: usize,
    referenced: bool,
    mutable: bool,
    dims: Vec<usize>,
    ty: Option<TypeExpr>,
    init: Option<CExpr>,
}

#[derive(Debug, Clone)]
struct CCallSite {
    name: String,
    args: Vec<CPlace>,
}

#[derive(Debug, Clone)]
struct CStack {
    name: String,
    slot: usize,
}

#[derive(Debug, Clone)]
struct CStmt {
    kind: CStmtKind,
    span: Range<usize>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
enum CStmtKind {
    Skip,
    Assert {
        condition: CExpr,
    },
    Seq(Vec<CStmt>),
    Update {
        target: CPlace,
        op: UpdateOp,
        expr: CExpr,
        alias_check: bool,
    },
    Swap {
        left: CPlace,
        right: CPlace,
    },
    Push {
        source: CPlace,
        stack: CStack,
    },
    Pop {
        target: CPlace,
        stack: CStack,
    },
    If {
        entry: CExpr,
        then_branch: Box<CStmt>,
        else_branch: Box<CStmt>,
        exit: CExpr,
    },
    Loop {
        entry: CExpr,
        body: Box<CStmt>,
        step: Box<CStmt>,
        exit: CExpr,
    },
    Iterate {
        name: String,
        slot: usize,
        start: CExpr,
        step: CExpr,
        end: CExpr,
        body: Box<CStmt>,
    },
    Call {
        name: String,
        target: Option<usize>,
        args: Vec<CPlace>,
        runtime_arg_check: bool,
        direct_frame: bool,
    },
    Uncall {
        name: String,
        target: Option<usize>,
        args: Vec<CPlace>,
        runtime_arg_check: bool,
        direct_frame: bool,
    },
    Read {
        target: CPlace,
    },
    Unread {
        target: CPlace,
    },
    Write {
        source: CPlace,
    },
    Unwrite {
        source: CPlace,
    },
    Show {
        targets: Vec<(String, usize)>,
    },
    Printf {
        format: String,
        args: Vec<CExpr>,
    },
    Declare {
        name: String,
        slot: usize,
        ty: TypeExpr,
        dims: Vec<usize>,
        init: Option<CExpr>,
    },
    Local {
        name: String,
        slot: usize,
        ty: Option<TypeExpr>,
        dims: Vec<Option<usize>>,
        refinement: Option<CExpr>,
        init: CExpr,
        body: Box<CStmt>,
        delocal_name: String,
        delocal_ty: Option<TypeExpr>,
        delocal_dims: Vec<Option<usize>>,
        delocal_refinement: Option<CExpr>,
        delocal: CExpr,
    },
}

#[derive(Debug, Clone)]
struct CPlace {
    name: String,
    slot: usize,
    indices: Vec<CExpr>,
}

#[derive(Debug, Clone)]
struct CExpr {
    kind: CExprKind,
    span: Range<usize>,
}

#[derive(Debug, Clone)]
enum CExprKind {
    Int(i64),
    Bool(bool),
    Array(Vec<CExpr>),
    Nil,
    Var {
        name: String,
        slot: usize,
    },
    Index {
        target: String,
        target_slot: usize,
        indices: Vec<CExpr>,
    },
    Empty {
        target: String,
        target_slot: usize,
    },
    Top {
        target: String,
        target_slot: usize,
    },
    Size {
        target: String,
        target_slot: usize,
    },
    Unary {
        op: UnaryOp,
        expr: Box<CExpr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<CExpr>,
        right: Box<CExpr>,
    },
}

#[derive(Debug, Clone, Default)]
struct SlotLayout {
    names: Vec<String>,
    indexes: BTreeMap<String, usize>,
}

impl SlotLayout {
    fn len(&self) -> usize {
        self.names.len()
    }

    fn ensure(&mut self, name: &str) -> usize {
        if let Some(slot) = self.indexes.get(name) {
            return *slot;
        }

        let slot = self.names.len();
        self.names.push(name.to_owned());
        self.indexes.insert(name.to_owned(), slot);
        slot
    }

    fn resolve(&self, name: &str) -> Result<usize, RuntimeError> {
        self.indexes
            .get(name)
            .copied()
            .ok_or_else(|| RuntimeError::new(format!("compiler missed variable `{name}`")))
    }
}

#[derive(Debug, Clone)]
struct SlotState {
    values: Vec<Option<Value>>,
}

struct Tapes<'a> {
    input: &'a mut VecDeque<Value>,
    output: &'a mut Vec<Value>,
    observations: &'a mut Vec<String>,
}

impl SlotState {
    fn empty(len: usize) -> Self {
        Self {
            values: vec![None; len],
        }
    }

    fn from_external(layout: &SlotLayout, state: State) -> Result<Self, RuntimeError> {
        let mut slot_state = Self::empty(layout.len());
        for (name, value) in state.store() {
            let slot = layout.resolve(name).map_err(|_| {
                RuntimeError::new(format!(
                    "compiled program was not prepared for variable `{name}`"
                ))
            })?;
            slot_state.set(slot, value.clone())?;
        }
        Ok(slot_state)
    }

    fn to_external(&self, layout: &SlotLayout) -> State {
        State::from_bindings(
            layout
                .names
                .iter()
                .zip(&self.values)
                .filter_map(|(name, value)| value.clone().map(|value| (name.clone(), value))),
        )
    }

    fn is_initialized(&self, slot: usize) -> bool {
        self.values.get(slot).is_some_and(Option::is_some)
    }

    fn value(&self, slot: usize, name: &str) -> Result<&Value, RuntimeError> {
        self.values
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| RuntimeError::new(format!("undefined variable `{name}`")))
    }

    fn value_mut(&mut self, slot: usize, name: &str) -> Result<&mut Value, RuntimeError> {
        self.values
            .get_mut(slot)
            .and_then(Option::as_mut)
            .ok_or_else(|| RuntimeError::new(format!("undefined variable `{name}`")))
    }

    fn set(&mut self, slot: usize, value: Value) -> Result<(), RuntimeError> {
        let Some(target) = self.values.get_mut(slot) else {
            return Err(RuntimeError::new("slot index out of bounds"));
        };
        *target = Some(value);
        Ok(())
    }

    fn clear(&mut self, slot: usize) -> Result<(), RuntimeError> {
        let Some(target) = self.values.get_mut(slot) else {
            return Err(RuntimeError::new("slot index out of bounds"));
        };
        *target = None;
        Ok(())
    }
}

fn initialize_globals(globals: &[CGlobal], state: &mut SlotState) -> Result<(), RuntimeError> {
    for global in globals {
        if state.is_initialized(global.slot) {
            let value = state.value(global.slot, &global.name)?;
            validate_declared_value(&global.name, &global.dims, global.ty.as_ref(), value)?;
            continue;
        }

        let value = initial_global_value(global, state)?;
        state.set(global.slot, value)?;
    }

    Ok(())
}

fn initial_global_value(global: &CGlobal, state: &SlotState) -> Result<Value, RuntimeError> {
    let value = if let Some(init) = &global.init {
        eval(init, state)?
    } else {
        default_value(&global.dims, global.ty.as_ref())?
    };
    validate_declared_value(&global.name, &global.dims, global.ty.as_ref(), &value)?;
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
    slot: usize,
    ty: &TypeExpr,
    dims: &[usize],
    init: Option<&CExpr>,
    state: &mut SlotState,
) -> Result<(), RuntimeError> {
    let value = if let Some(init) = init {
        eval(init, state)?
    } else {
        default_value(dims, Some(ty))?
    };
    validate_declared_shape(name, dims, &value)?;

    if state.is_initialized(slot) {
        let existing = state.value(slot, name)?;
        validate_declared_value(name, dims, Some(ty), existing)?;
        if existing == &value {
            return Ok(());
        }

        return Err(RuntimeError::new(format!(
            "declaration `{name}` expected {value}, found {existing}"
        )));
    }

    state.set(slot, value)
}

#[derive(Debug, Clone, Copy)]
enum Direction {
    Call,
    Uncall,
}

impl Direction {
    fn gerund(self) -> &'static str {
        match self {
            Self::Call => "calling",
            Self::Uncall => "uncalling",
        }
    }
}

fn compile_procedure(
    procedure: &Proc,
    globals: &[GlobalDecl],
    layout: SlotLayout,
) -> Result<CompiledProcedure, RuntimeError> {
    let inverse_body = reverie_core::invert_stmt(&procedure.body);
    let body = compile_stmt(&procedure.body, &layout)?;
    let inverse_body = compile_stmt(&inverse_body, &layout)?;
    let mut mutable_roots = Vec::new();
    let mut referenced_roots = Vec::new();
    let mut call_sites = Vec::new();
    let mut call_targets = Vec::new();
    collect_mutated_roots(&body, &mut mutable_roots);
    collect_mutated_roots(&inverse_body, &mut mutable_roots);
    collect_referenced_roots(&body, &mut referenced_roots);
    collect_referenced_roots(&inverse_body, &mut referenced_roots);
    for param in &procedure.params {
        if let Some(refinement) = &param.refinement {
            let refinement = compile_expr(refinement, &layout)?;
            collect_expr_roots(&refinement, &mut referenced_roots);
        }
    }
    collect_call_sites(&body, &mut call_sites);
    collect_call_sites(&inverse_body, &mut call_sites);
    collect_call_targets(&body, &mut call_targets);
    collect_call_targets(&inverse_body, &mut call_targets);
    Ok(CompiledProcedure {
        name: procedure.name.clone(),
        mutable_param_indices: Vec::new(),
        active_global_indices: Vec::new(),
        direct_frame_eligible: false,
        has_param_refinements: procedure
            .params
            .iter()
            .any(|param| param.refinement.is_some()),
        params: procedure
            .params
            .iter()
            .map(|param| {
                Ok(CParam {
                    name: param.name.clone(),
                    slot: layout.resolve(&param.name)?,
                    mutable: mutable_roots.iter().any(|name| name == &param.name),
                    refinement: param
                        .refinement
                        .as_ref()
                        .map(|expr| compile_expr(expr, &layout))
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>, RuntimeError>>()?,
        globals: globals
            .iter()
            .map(|global| {
                Ok(CGlobal {
                    name: global.name.clone(),
                    slot: layout.resolve(&global.name)?,
                    referenced: referenced_roots.iter().any(|name| name == &global.name),
                    mutable: mutable_roots.iter().any(|name| name == &global.name),
                    dims: global.dims.clone(),
                    ty: global.ty.as_ref().map(|ty| ty.node.clone()),
                    init: global
                        .init
                        .as_ref()
                        .map(|expr| compile_expr(expr, &layout))
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>, RuntimeError>>()?,
        body,
        inverse_body,
        call_sites,
        call_targets,
        layout,
        span: procedure.span.clone(),
    })
}

fn compile_stmt(statement: &SpannedStmt, layout: &SlotLayout) -> Result<CStmt, RuntimeError> {
    let kind = match &statement.node {
        Stmt::Skip => CStmtKind::Skip,
        Stmt::Assert { condition } => CStmtKind::Assert {
            condition: compile_expr(condition, layout)?,
        },
        Stmt::Seq(statements) => CStmtKind::Seq(
            statements
                .iter()
                .map(|statement| compile_stmt(statement, layout))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Stmt::Update { target, op, expr } => {
            let target = compile_place(target, layout)?;
            let expr = compile_expr(expr, layout)?;
            let alias_check = expr_may_alias_place(&target, &expr);
            CStmtKind::Update {
                alias_check,
                target,
                op: *op,
                expr,
            }
        }
        Stmt::Swap { left, right } => CStmtKind::Swap {
            left: compile_place(left, layout)?,
            right: compile_place(right, layout)?,
        },
        Stmt::Push { source, stack } => CStmtKind::Push {
            source: compile_place(source, layout)?,
            stack: compile_stack(stack, layout)?,
        },
        Stmt::Pop { target, stack } => CStmtKind::Pop {
            target: compile_place(target, layout)?,
            stack: compile_stack(stack, layout)?,
        },
        Stmt::If {
            entry,
            then_branch,
            else_branch,
            exit,
        } => CStmtKind::If {
            entry: compile_expr(entry, layout)?,
            then_branch: Box::new(compile_stmt(then_branch, layout)?),
            else_branch: Box::new(compile_stmt(else_branch, layout)?),
            exit: compile_expr(exit, layout)?,
        },
        Stmt::Loop {
            entry,
            body,
            step,
            exit,
        } => CStmtKind::Loop {
            entry: compile_expr(entry, layout)?,
            body: Box::new(compile_stmt(body, layout)?),
            step: Box::new(compile_stmt(step, layout)?),
            exit: compile_expr(exit, layout)?,
        },
        Stmt::Iterate {
            name,
            start,
            step,
            end,
            body,
        } => CStmtKind::Iterate {
            name: name.clone(),
            slot: layout.resolve(name)?,
            start: compile_expr(start, layout)?,
            step: compile_expr(step, layout)?,
            end: compile_expr(end, layout)?,
            body: Box::new(compile_stmt(body, layout)?),
        },
        Stmt::Call { name, args } => {
            let (args, runtime_arg_check) = compile_call_args(name, args, layout)?;
            CStmtKind::Call {
                name: name.clone(),
                target: None,
                args,
                runtime_arg_check,
                direct_frame: false,
            }
        }
        Stmt::Uncall { name, args } => {
            let (args, runtime_arg_check) = compile_call_args(name, args, layout)?;
            CStmtKind::Uncall {
                name: name.clone(),
                target: None,
                args,
                runtime_arg_check,
                direct_frame: false,
            }
        }
        Stmt::Read { target } => CStmtKind::Read {
            target: compile_place(target, layout)?,
        },
        Stmt::Unread { target } => CStmtKind::Unread {
            target: compile_place(target, layout)?,
        },
        Stmt::Write { source } => CStmtKind::Write {
            source: compile_place(source, layout)?,
        },
        Stmt::Unwrite { source } => CStmtKind::Unwrite {
            source: compile_place(source, layout)?,
        },
        Stmt::Show { targets } => CStmtKind::Show {
            targets: targets
                .iter()
                .map(|target| Ok((target.clone(), layout.resolve(target)?)))
                .collect::<Result<Vec<_>, RuntimeError>>()?,
        },
        Stmt::Printf { format, args } => CStmtKind::Printf {
            format: format.clone(),
            args: args
                .iter()
                .map(|arg| compile_expr(arg, layout))
                .collect::<Result<Vec<_>, _>>()?,
        },
        Stmt::Declare {
            name,
            ty,
            dims,
            init,
            ..
        } => CStmtKind::Declare {
            name: name.clone(),
            slot: layout.resolve(name)?,
            ty: ty.node.clone(),
            dims: dims.clone(),
            init: init
                .as_ref()
                .map(|expr| compile_expr(expr, layout))
                .transpose()?,
        },
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
        } => CStmtKind::Local {
            name: name.clone(),
            slot: layout.resolve(name)?,
            ty: ty.as_ref().map(|ty| ty.node.clone()),
            dims: dims.clone(),
            refinement: refinement
                .as_ref()
                .map(|expr| compile_expr(expr, layout))
                .transpose()?,
            init: compile_expr(init, layout)?,
            body: Box::new(compile_stmt(body, layout)?),
            delocal_name: delocal_name.clone(),
            delocal_ty: delocal_ty.as_ref().map(|ty| ty.node.clone()),
            delocal_dims: delocal_dims.clone(),
            delocal_refinement: delocal_refinement
                .as_ref()
                .map(|expr| compile_expr(expr, layout))
                .transpose()?,
            delocal: compile_expr(delocal, layout)?,
        },
    };

    Ok(CStmt {
        kind,
        span: statement.span.clone(),
    })
}

fn compile_args(args: &[Place], layout: &SlotLayout) -> Result<Vec<CPlace>, RuntimeError> {
    args.iter().map(|arg| compile_place(arg, layout)).collect()
}

fn compile_call_args(
    name: &str,
    args: &[Place],
    layout: &SlotLayout,
) -> Result<(Vec<CPlace>, bool), RuntimeError> {
    let args = compile_args(args, layout)?;
    let runtime_arg_check = call_args_need_runtime_check(&args);
    if !runtime_arg_check {
        ensure_unique_args(name, &args)?;
    }
    Ok((args, runtime_arg_check))
}

fn call_args_need_runtime_check(args: &[CPlace]) -> bool {
    args.iter().any(|arg| {
        arg.indices
            .iter()
            .any(|index| const_int_value(index).is_none())
    })
}

fn compile_place(place: &Place, layout: &SlotLayout) -> Result<CPlace, RuntimeError> {
    Ok(CPlace {
        name: place.name.clone(),
        slot: layout.resolve(&place.name)?,
        indices: place
            .indices
            .iter()
            .map(|index| compile_expr(index, layout))
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn collect_mutated_roots(statement: &CStmt, roots: &mut Vec<String>) {
    match &statement.kind {
        CStmtKind::Skip
        | CStmtKind::Assert { .. }
        | CStmtKind::Write { .. }
        | CStmtKind::Unwrite { .. }
        | CStmtKind::Show { .. }
        | CStmtKind::Printf { .. } => {}
        CStmtKind::Seq(statements) => {
            for statement in statements {
                collect_mutated_roots(statement, roots);
            }
        }
        CStmtKind::Update { target, .. }
        | CStmtKind::Read { target }
        | CStmtKind::Unread { target } => push_unique_root(roots, &target.name),
        CStmtKind::Swap { left, right } => {
            push_unique_root(roots, &left.name);
            push_unique_root(roots, &right.name);
        }
        CStmtKind::Push { source, stack } => {
            push_unique_root(roots, &source.name);
            push_unique_root(roots, &stack.name);
        }
        CStmtKind::Pop { target, stack } => {
            push_unique_root(roots, &target.name);
            push_unique_root(roots, &stack.name);
        }
        CStmtKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_mutated_roots(then_branch, roots);
            collect_mutated_roots(else_branch, roots);
        }
        CStmtKind::Loop { body, step, .. } => {
            collect_mutated_roots(body, roots);
            collect_mutated_roots(step, roots);
        }
        CStmtKind::Iterate { name, body, .. } => {
            push_unique_root(roots, name);
            collect_mutated_roots(body, roots);
        }
        CStmtKind::Call { .. } | CStmtKind::Uncall { .. } => {}
        CStmtKind::Declare { name, .. } => push_unique_root(roots, name),
        CStmtKind::Local { name, body, .. } => {
            push_unique_root(roots, name);
            collect_mutated_roots(body, roots);
        }
    }
}

fn collect_referenced_roots(statement: &CStmt, roots: &mut Vec<String>) {
    match &statement.kind {
        CStmtKind::Skip => {}
        CStmtKind::Assert { condition } => collect_expr_roots(condition, roots),
        CStmtKind::Seq(statements) => {
            for statement in statements {
                collect_referenced_roots(statement, roots);
            }
        }
        CStmtKind::Update { target, expr, .. } => {
            collect_place_roots(target, roots);
            collect_expr_roots(expr, roots);
        }
        CStmtKind::Swap { left, right } => {
            collect_place_roots(left, roots);
            collect_place_roots(right, roots);
        }
        CStmtKind::Push { source, stack } => {
            collect_place_roots(source, roots);
            push_unique_root(roots, &stack.name);
        }
        CStmtKind::Pop { target, stack } => {
            collect_place_roots(target, roots);
            push_unique_root(roots, &stack.name);
        }
        CStmtKind::If {
            entry,
            then_branch,
            else_branch,
            exit,
        } => {
            collect_expr_roots(entry, roots);
            collect_referenced_roots(then_branch, roots);
            collect_referenced_roots(else_branch, roots);
            collect_expr_roots(exit, roots);
        }
        CStmtKind::Loop {
            entry,
            body,
            step,
            exit,
        } => {
            collect_expr_roots(entry, roots);
            collect_referenced_roots(body, roots);
            collect_referenced_roots(step, roots);
            collect_expr_roots(exit, roots);
        }
        CStmtKind::Iterate {
            name,
            start,
            step,
            end,
            body,
            ..
        } => {
            push_unique_root(roots, name);
            collect_expr_roots(start, roots);
            collect_expr_roots(step, roots);
            collect_expr_roots(end, roots);
            collect_referenced_roots(body, roots);
        }
        CStmtKind::Call { args, .. } | CStmtKind::Uncall { args, .. } => {
            for arg in args {
                collect_place_roots(arg, roots);
            }
        }
        CStmtKind::Read { target } | CStmtKind::Unread { target } => {
            collect_place_roots(target, roots)
        }
        CStmtKind::Write { source } | CStmtKind::Unwrite { source } => {
            collect_place_roots(source, roots);
        }
        CStmtKind::Show { targets } => {
            for (name, _) in targets {
                push_unique_root(roots, name);
            }
        }
        CStmtKind::Printf { args, .. } => {
            for arg in args {
                collect_expr_roots(arg, roots);
            }
        }
        CStmtKind::Declare { name, init, .. } => {
            push_unique_root(roots, name);
            if let Some(init) = init {
                collect_expr_roots(init, roots);
            }
        }
        CStmtKind::Local {
            name,
            refinement,
            init,
            body,
            delocal_name,
            delocal_refinement,
            delocal,
            ..
        } => {
            push_unique_root(roots, name);
            collect_expr_roots(init, roots);
            if let Some(refinement) = refinement {
                collect_expr_roots(refinement, roots);
            }
            collect_referenced_roots(body, roots);
            push_unique_root(roots, delocal_name);
            if let Some(delocal_refinement) = delocal_refinement {
                collect_expr_roots(delocal_refinement, roots);
            }
            collect_expr_roots(delocal, roots);
        }
    }
}

fn collect_place_roots(place: &CPlace, roots: &mut Vec<String>) {
    push_unique_root(roots, &place.name);
    for index in &place.indices {
        collect_expr_roots(index, roots);
    }
}

fn collect_expr_roots(expr: &CExpr, roots: &mut Vec<String>) {
    match &expr.kind {
        CExprKind::Int(_) | CExprKind::Bool(_) | CExprKind::Nil => {}
        CExprKind::Array(elements) => {
            for element in elements {
                collect_expr_roots(element, roots);
            }
        }
        CExprKind::Var { name, .. }
        | CExprKind::Empty { target: name, .. }
        | CExprKind::Top { target: name, .. }
        | CExprKind::Size { target: name, .. } => push_unique_root(roots, name),
        CExprKind::Index {
            target, indices, ..
        } => {
            push_unique_root(roots, target);
            for index in indices {
                collect_expr_roots(index, roots);
            }
        }
        CExprKind::Unary { expr, .. } => collect_expr_roots(expr, roots),
        CExprKind::Binary { left, right, .. } => {
            collect_expr_roots(left, roots);
            collect_expr_roots(right, roots);
        }
    }
}

fn collect_call_sites(statement: &CStmt, sites: &mut Vec<CCallSite>) {
    match &statement.kind {
        CStmtKind::Call { name, args, .. } | CStmtKind::Uncall { name, args, .. } => {
            sites.push(CCallSite {
                name: name.clone(),
                args: args.clone(),
            });
        }
        CStmtKind::Seq(statements) => {
            for statement in statements {
                collect_call_sites(statement, sites);
            }
        }
        CStmtKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_call_sites(then_branch, sites);
            collect_call_sites(else_branch, sites);
        }
        CStmtKind::Loop { body, step, .. } => {
            collect_call_sites(body, sites);
            collect_call_sites(step, sites);
        }
        CStmtKind::Iterate { body, .. } | CStmtKind::Local { body, .. } => {
            collect_call_sites(body, sites);
        }
        CStmtKind::Skip
        | CStmtKind::Assert { .. }
        | CStmtKind::Update { .. }
        | CStmtKind::Swap { .. }
        | CStmtKind::Push { .. }
        | CStmtKind::Pop { .. }
        | CStmtKind::Read { .. }
        | CStmtKind::Unread { .. }
        | CStmtKind::Write { .. }
        | CStmtKind::Unwrite { .. }
        | CStmtKind::Show { .. }
        | CStmtKind::Printf { .. }
        | CStmtKind::Declare { .. } => {}
    }
}

fn collect_call_targets(statement: &CStmt, targets: &mut Vec<String>) {
    match &statement.kind {
        CStmtKind::Call { name, .. } | CStmtKind::Uncall { name, .. } => {
            push_unique_root(targets, name)
        }
        CStmtKind::Seq(statements) => {
            for statement in statements {
                collect_call_targets(statement, targets);
            }
        }
        CStmtKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_call_targets(then_branch, targets);
            collect_call_targets(else_branch, targets);
        }
        CStmtKind::Loop { body, step, .. } => {
            collect_call_targets(body, targets);
            collect_call_targets(step, targets);
        }
        CStmtKind::Iterate { body, .. } | CStmtKind::Local { body, .. } => {
            collect_call_targets(body, targets);
        }
        CStmtKind::Skip
        | CStmtKind::Assert { .. }
        | CStmtKind::Update { .. }
        | CStmtKind::Swap { .. }
        | CStmtKind::Push { .. }
        | CStmtKind::Pop { .. }
        | CStmtKind::Read { .. }
        | CStmtKind::Unread { .. }
        | CStmtKind::Write { .. }
        | CStmtKind::Unwrite { .. }
        | CStmtKind::Show { .. }
        | CStmtKind::Printf { .. }
        | CStmtKind::Declare { .. } => {}
    }
}

fn propagate_procedure_global_effects(
    procedures: &mut [CompiledProcedure],
    globals: &[GlobalDecl],
) {
    let global_names = globals
        .iter()
        .map(|global| global.name.as_str())
        .collect::<Vec<_>>();

    loop {
        let snapshot = procedures
            .iter()
            .map(|procedure| {
                (
                    procedure.name.clone(),
                    procedure
                        .globals
                        .iter()
                        .map(|global| (global.name.clone(), global.referenced, global.mutable))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<HashMap<_, _>>();

        let mut changed = false;
        for procedure in procedures.iter_mut() {
            let call_targets = procedure.call_targets.clone();
            for callee_name in call_targets {
                let Some(callee_globals) = snapshot.get(&callee_name) else {
                    continue;
                };

                for global_name in &global_names {
                    let Some((_, callee_referenced, callee_mutable)) = callee_globals
                        .iter()
                        .find(|(name, _, _)| name == global_name)
                    else {
                        continue;
                    };

                    let Some(global) = procedure
                        .globals
                        .iter_mut()
                        .find(|global| global.name == *global_name)
                    else {
                        continue;
                    };

                    if *callee_referenced && !global.referenced {
                        global.referenced = true;
                        changed = true;
                    }
                    if *callee_mutable && !global.mutable {
                        global.mutable = true;
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }
}

fn propagate_procedure_param_effects(procedures: &mut [CompiledProcedure], globals: &[GlobalDecl]) {
    loop {
        let snapshot = procedures
            .iter()
            .map(|procedure| {
                (
                    procedure.name.clone(),
                    procedure
                        .params
                        .iter()
                        .map(|param| (param.name.clone(), param.mutable))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<HashMap<_, _>>();

        let mut changed = false;
        for procedure in procedures.iter_mut() {
            let call_sites = procedure.call_sites.clone();
            for call_site in call_sites {
                let Some(callee_params) = snapshot.get(&call_site.name) else {
                    continue;
                };

                for (arg, (_, callee_mutable)) in call_site.args.iter().zip(callee_params) {
                    if !callee_mutable {
                        continue;
                    }

                    if let Some(param) = procedure
                        .params
                        .iter_mut()
                        .find(|param| param.name == arg.name)
                        && !param.mutable
                    {
                        param.mutable = true;
                        changed = true;
                    }

                    if globals.iter().any(|global| global.name == arg.name)
                        && let Some(global) = procedure
                            .globals
                            .iter_mut()
                            .find(|global| global.name == arg.name)
                        && !global.mutable
                    {
                        global.mutable = true;
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }
}

fn refresh_procedure_runtime_indices(procedures: &mut [CompiledProcedure]) {
    for procedure in procedures {
        procedure.mutable_param_indices = procedure
            .params
            .iter()
            .enumerate()
            .filter_map(|(index, param)| param.mutable.then_some(index))
            .collect();
        procedure.active_global_indices = procedure
            .globals
            .iter()
            .enumerate()
            .filter_map(|(index, global)| {
                if !(global.referenced || global.mutable) {
                    return None;
                }
                if procedure
                    .params
                    .iter()
                    .any(|param| param.slot == global.slot)
                {
                    return None;
                }
                Some(index)
            })
            .collect();
        procedure.direct_frame_eligible = procedure_has_no_local_slots(procedure);
    }
}

fn procedure_has_no_local_slots(procedure: &CompiledProcedure) -> bool {
    let mut covered_slots = vec![false; procedure.layout.len()];
    for global in &procedure.globals {
        if let Some(covered) = covered_slots.get_mut(global.slot) {
            *covered = true;
        }
    }
    for param in &procedure.params {
        if let Some(covered) = covered_slots.get_mut(param.slot) {
            *covered = true;
        }
    }
    covered_slots.into_iter().all(|covered| covered)
}

fn refresh_procedure_call_sites(
    procedures: &mut [CompiledProcedure],
    procedure_indices: &HashMap<String, usize>,
    signatures: &HashMap<String, DirectFrameSignature>,
) -> Result<(), RuntimeError> {
    for procedure in procedures {
        refresh_call_metadata(&mut procedure.body, procedure_indices, signatures)?;
        refresh_call_metadata(&mut procedure.inverse_body, procedure_indices, signatures)?;
    }
    Ok(())
}

fn direct_frame_signatures(
    procedures: &[CompiledProcedure],
) -> HashMap<String, DirectFrameSignature> {
    procedures
        .iter()
        .map(|procedure| (procedure.name.clone(), procedure.direct_frame_signature()))
        .collect()
}

fn refresh_call_metadata(
    statement: &mut CStmt,
    procedure_indices: &HashMap<String, usize>,
    signatures: &HashMap<String, DirectFrameSignature>,
) -> Result<(), RuntimeError> {
    match &mut statement.kind {
        CStmtKind::Call {
            name,
            target,
            args,
            runtime_arg_check,
            direct_frame,
        }
        | CStmtKind::Uncall {
            name,
            target,
            args,
            runtime_arg_check,
            direct_frame,
        } => {
            *target = Some(
                *procedure_indices
                    .get(name)
                    .ok_or_else(|| RuntimeError::new(format!("unknown procedure `{name}`")))?,
            );
            *direct_frame = !*runtime_arg_check
                && signatures
                    .get(name)
                    .is_some_and(|signature| direct_frame_matches(signature, args));
        }
        CStmtKind::Seq(statements) => {
            for statement in statements {
                refresh_call_metadata(statement, procedure_indices, signatures)?;
            }
        }
        CStmtKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            refresh_call_metadata(then_branch, procedure_indices, signatures)?;
            refresh_call_metadata(else_branch, procedure_indices, signatures)?;
        }
        CStmtKind::Loop { body, step, .. } => {
            refresh_call_metadata(body, procedure_indices, signatures)?;
            refresh_call_metadata(step, procedure_indices, signatures)?;
        }
        CStmtKind::Iterate { body, .. } | CStmtKind::Local { body, .. } => {
            refresh_call_metadata(body, procedure_indices, signatures)?;
        }
        CStmtKind::Skip
        | CStmtKind::Assert { .. }
        | CStmtKind::Update { .. }
        | CStmtKind::Swap { .. }
        | CStmtKind::Push { .. }
        | CStmtKind::Pop { .. }
        | CStmtKind::Read { .. }
        | CStmtKind::Unread { .. }
        | CStmtKind::Write { .. }
        | CStmtKind::Unwrite { .. }
        | CStmtKind::Show { .. }
        | CStmtKind::Printf { .. }
        | CStmtKind::Declare { .. } => {}
    }
    Ok(())
}

fn direct_frame_matches(signature: &DirectFrameSignature, args: &[CPlace]) -> bool {
    signature.eligible
        && signature.param_slots.len() == args.len()
        && signature
            .param_slots
            .iter()
            .zip(args)
            .all(|(slot, arg)| arg.indices.is_empty() && *slot == arg.slot)
}

fn push_unique_root(roots: &mut Vec<String>, name: &str) {
    if !roots.iter().any(|root| root == name) {
        roots.push(name.to_owned());
    }
}

fn compile_stack(stack: &str, layout: &SlotLayout) -> Result<CStack, RuntimeError> {
    Ok(CStack {
        name: stack.to_owned(),
        slot: layout.resolve(stack)?,
    })
}

fn compile_expr(expr: &SpannedExpr, layout: &SlotLayout) -> Result<CExpr, RuntimeError> {
    let kind = match &expr.node {
        Expr::Int { value, .. } => CExprKind::Int(*value),
        Expr::Bool(value) => CExprKind::Bool(*value),
        Expr::Array(elements) => CExprKind::Array(
            elements
                .iter()
                .map(|element| compile_expr(element, layout))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Expr::Nil => CExprKind::Nil,
        Expr::Var(name) => CExprKind::Var {
            name: name.clone(),
            slot: layout.resolve(name)?,
        },
        Expr::Index { target, indices } => CExprKind::Index {
            target: target.clone(),
            target_slot: layout.resolve(target)?,
            indices: indices
                .iter()
                .map(|index| compile_expr(index, layout))
                .collect::<Result<Vec<_>, _>>()?,
        },
        Expr::Empty { target } => CExprKind::Empty {
            target: target.clone(),
            target_slot: layout.resolve(target)?,
        },
        Expr::Top { target } => CExprKind::Top {
            target: target.clone(),
            target_slot: layout.resolve(target)?,
        },
        Expr::Size { target } => CExprKind::Size {
            target: target.clone(),
            target_slot: layout.resolve(target)?,
        },
        Expr::Unary { op, expr } => CExprKind::Unary {
            op: *op,
            expr: Box::new(compile_expr(expr, layout)?),
        },
        Expr::Binary { op, left, right } => CExprKind::Binary {
            op: *op,
            left: Box::new(compile_expr(left, layout)?),
            right: Box::new(compile_expr(right, layout)?),
        },
    };

    Ok(CExpr {
        kind,
        span: expr.span.clone(),
    })
}

fn collect_stmt_names(statement: &SpannedStmt, layout: &mut SlotLayout) {
    match &statement.node {
        Stmt::Skip => {}
        Stmt::Assert { condition } => {
            collect_expr_names(condition, layout);
        }
        Stmt::Seq(statements) => {
            for statement in statements {
                collect_stmt_names(statement, layout);
            }
        }
        Stmt::Update { target, expr, .. } => {
            collect_place_names(target, layout);
            collect_expr_names(expr, layout);
        }
        Stmt::Swap { left, right } => {
            collect_place_names(left, layout);
            collect_place_names(right, layout);
        }
        Stmt::Push { source, stack } => {
            collect_place_names(source, layout);
            layout.ensure(stack);
        }
        Stmt::Pop { target, stack } => {
            collect_place_names(target, layout);
            layout.ensure(stack);
        }
        Stmt::If {
            entry,
            then_branch,
            else_branch,
            exit,
        } => {
            collect_expr_names(entry, layout);
            collect_stmt_names(then_branch, layout);
            collect_stmt_names(else_branch, layout);
            collect_expr_names(exit, layout);
        }
        Stmt::Loop {
            entry,
            body,
            step,
            exit,
        } => {
            collect_expr_names(entry, layout);
            collect_stmt_names(body, layout);
            collect_stmt_names(step, layout);
            collect_expr_names(exit, layout);
        }
        Stmt::Iterate {
            name,
            start,
            step,
            end,
            body,
        } => {
            layout.ensure(name);
            collect_expr_names(start, layout);
            collect_expr_names(step, layout);
            collect_expr_names(end, layout);
            collect_stmt_names(body, layout);
        }
        Stmt::Call { args, .. } | Stmt::Uncall { args, .. } => {
            for arg in args {
                collect_place_names(arg, layout);
            }
        }
        Stmt::Read { target } | Stmt::Unread { target } => {
            collect_place_names(target, layout);
        }
        Stmt::Write { source } | Stmt::Unwrite { source } => {
            collect_place_names(source, layout);
        }
        Stmt::Show { targets } => {
            for target in targets {
                layout.ensure(target);
            }
        }
        Stmt::Printf { args, .. } => {
            for arg in args {
                collect_expr_names(arg, layout);
            }
        }
        Stmt::Declare { name, init, .. } => {
            layout.ensure(name);
            if let Some(init) = init {
                collect_expr_names(init, layout);
            }
        }
        Stmt::Local {
            name,
            refinement,
            init,
            body,
            delocal_name,
            delocal_refinement,
            delocal,
            ..
        } => {
            layout.ensure(name);
            layout.ensure(delocal_name);
            if let Some(refinement) = refinement {
                collect_expr_names(refinement, layout);
            }
            if let Some(delocal_refinement) = delocal_refinement {
                collect_expr_names(delocal_refinement, layout);
            }
            collect_expr_names(init, layout);
            collect_stmt_names(body, layout);
            collect_expr_names(delocal, layout);
        }
    }
}

fn collect_place_names(place: &Place, layout: &mut SlotLayout) {
    layout.ensure(&place.name);
    for index in &place.indices {
        collect_expr_names(index, layout);
    }
}

fn collect_expr_names(expr: &SpannedExpr, layout: &mut SlotLayout) {
    match &expr.node {
        Expr::Int { .. } | Expr::Bool(_) | Expr::Nil => {}
        Expr::Array(elements) => {
            for element in elements {
                collect_expr_names(element, layout);
            }
        }
        Expr::Var(name) => {
            layout.ensure(name);
        }
        Expr::Index { target, indices } => {
            layout.ensure(target);
            for index in indices {
                collect_expr_names(index, layout);
            }
        }
        Expr::Empty { target } | Expr::Top { target } => {
            layout.ensure(target);
        }
        Expr::Size { target } => {
            layout.ensure(target);
        }
        Expr::Unary { expr, .. } => {
            collect_expr_names(expr, layout);
        }
        Expr::Binary { left, right, .. } => {
            collect_expr_names(left, layout);
            collect_expr_names(right, layout);
        }
    }
}

fn expr_may_alias_place(place: &CPlace, expr: &CExpr) -> bool {
    match &expr.kind {
        CExprKind::Int(_) | CExprKind::Bool(_) | CExprKind::Nil | CExprKind::Size { .. } => false,
        CExprKind::Array(elements) => elements
            .iter()
            .any(|element| expr_may_alias_place(place, element)),
        CExprKind::Var {
            name: read_name, ..
        } => read_name == &place.name,
        CExprKind::Index {
            target, indices, ..
        } => {
            indices
                .iter()
                .any(|index| expr_may_alias_place(place, index))
                || indexed_read_may_alias_place(place, target, indices)
        }
        CExprKind::Empty { target, .. } | CExprKind::Top { target, .. } => target == &place.name,
        CExprKind::Unary { expr, .. } => expr_may_alias_place(place, expr),
        CExprKind::Binary { left, right, .. } => {
            expr_may_alias_place(place, left) || expr_may_alias_place(place, right)
        }
    }
}

fn indexed_read_may_alias_place(place: &CPlace, read_target: &str, read_indices: &[CExpr]) -> bool {
    if read_target != place.name {
        return false;
    }
    if place.indices.is_empty() {
        return true;
    }
    if place.indices.len() != read_indices.len() {
        return true;
    }
    !constant_indices_proven_distinct(&place.indices, read_indices)
}

fn constant_indices_proven_distinct(left: &[CExpr], right: &[CExpr]) -> bool {
    left.iter().zip(right).any(|(left, right)| {
        match (const_int_value(left), const_int_value(right)) {
            (Some(left), Some(right)) => left != right,
            _ => false,
        }
    })
}

fn const_int_value(expr: &CExpr) -> Option<i64> {
    match &expr.kind {
        CExprKind::Int(value) => Some(*value),
        CExprKind::Unary {
            op: UnaryOp::Neg,
            expr,
        } => const_int_value(expr).map(i64::wrapping_neg),
        CExprKind::Binary { op, left, right } => {
            let left = const_int_value(left)?;
            let right = const_int_value(right)?;
            match op {
                BinaryOp::Add => Some(left.wrapping_add(right)),
                BinaryOp::Sub => Some(left.wrapping_sub(right)),
                BinaryOp::Mul => Some(left.wrapping_mul(right)),
                BinaryOp::FixedMul => Some(((i128::from(left) * i128::from(right)) >> 31) as i64),
                BinaryOp::Div if right != 0 => Some(left.wrapping_div(right)),
                BinaryOp::Rem if right != 0 => Some(left.wrapping_rem(right)),
                BinaryOp::Shl => Some(left.wrapping_shl(right as u32)),
                BinaryOp::Shr => Some(left.wrapping_shr(right as u32)),
                BinaryOp::BitAnd => Some(left & right),
                BinaryOp::BitXor => Some(left ^ right),
                BinaryOp::BitOr => Some(left | right),
                BinaryOp::Div
                | BinaryOp::Rem
                | BinaryOp::Eq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::LtEq
                | BinaryOp::Gt
                | BinaryOp::GtEq
                | BinaryOp::And
                | BinaryOp::Or => None,
            }
        }
        CExprKind::Bool(_)
        | CExprKind::Array(_)
        | CExprKind::Nil
        | CExprKind::Var { .. }
        | CExprKind::Index { .. }
        | CExprKind::Empty { .. }
        | CExprKind::Top { .. }
        | CExprKind::Size { .. } => None,
        CExprKind::Unary { .. } => None,
    }
}

fn eval(expr: &CExpr, state: &SlotState) -> Result<Value, RuntimeError> {
    match &expr.kind {
        CExprKind::Int(value) => Ok(Value::Int(*value)),
        CExprKind::Bool(value) => Ok(Value::Bool(*value)),
        CExprKind::Array(elements) => elements
            .iter()
            .map(|element| eval(element, state))
            .collect::<Result<Vec<_>, _>>()
            .and_then(array_value),
        CExprKind::Nil => Ok(Value::Stack(Vec::new())),
        CExprKind::Var { name, slot } => state
            .value(*slot, name)
            .cloned()
            .map_err(|error| error.with_span_if_missing(expr.span.clone())),
        CExprKind::Index {
            target,
            target_slot,
            indices,
        } => place_value_parts(target, *target_slot, indices, state)
            .cloned()
            .map_err(|error| error.with_span_if_missing(expr.span.clone())),
        CExprKind::Empty {
            target,
            target_slot,
        } => eval_empty(target, *target_slot, expr.span.clone(), state),
        CExprKind::Top {
            target,
            target_slot,
        } => eval_top(target, *target_slot, expr.span.clone(), state),
        CExprKind::Size {
            target,
            target_slot,
        } => eval_size(target, *target_slot, expr.span.clone(), state),
        CExprKind::Unary { op, expr } => eval_unary(*op, expr, state),
        CExprKind::Binary { op, left, right } => eval_binary(*op, left, right, state),
    }
}

fn eval_empty(
    target: &str,
    target_slot: usize,
    span: Range<usize>,
    state: &SlotState,
) -> Result<Value, RuntimeError> {
    let stack = expect_stack(target, target_slot, span, state)?;
    Ok(Value::Bool(stack.is_empty()))
}

fn eval_top(
    target: &str,
    target_slot: usize,
    span: Range<usize>,
    state: &SlotState,
) -> Result<Value, RuntimeError> {
    let stack = expect_stack(target, target_slot, span.clone(), state)?;
    stack
        .first()
        .copied()
        .map(Value::Int)
        .ok_or_else(|| RuntimeError::at(format!("top({target}) expected a non-empty stack"), span))
}

fn eval_size(
    target: &str,
    target_slot: usize,
    span: Range<usize>,
    state: &SlotState,
) -> Result<Value, RuntimeError> {
    match state
        .value(target_slot, target)
        .map_err(|error| error.with_span_if_missing(span.clone()))?
    {
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

fn eval_unary(op: UnaryOp, expr: &CExpr, state: &SlotState) -> Result<Value, RuntimeError> {
    Ok(match op {
        UnaryOp::Neg => Value::Int(eval_int(expr, state)?.wrapping_neg()),
        UnaryOp::Not => Value::Bool(!eval_bool(expr, state)?),
    })
}

fn eval_binary(
    op: BinaryOp,
    left: &CExpr,
    right: &CExpr,
    state: &SlotState,
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
        BinaryOp::Eq => Value::Bool(eval_eq(left, right, state)?),
        BinaryOp::NotEq => Value::Bool(!eval_eq(left, right, state)?),
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
    left: &CExpr,
    right: &CExpr,
    state: &SlotState,
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

fn eval_eq(left: &CExpr, right: &CExpr, state: &SlotState) -> Result<bool, RuntimeError> {
    if let Some(result) = eval_literal_eq(left, right, state) {
        return result;
    }
    if let Some(result) = eval_literal_eq(right, left, state) {
        return result;
    }

    match (expr_value_ref(left, state), expr_value_ref(right, state)) {
        (Some(left), Some(right)) => Ok(left? == right?),
        (Some(left), None) => Ok(left? == &eval(right, state)?),
        (None, Some(right)) => Ok(&eval(left, state)? == right?),
        (None, None) => Ok(eval(left, state)? == eval(right, state)?),
    }
}

fn eval_literal_eq(
    value_expr: &CExpr,
    literal_expr: &CExpr,
    state: &SlotState,
) -> Option<Result<bool, RuntimeError>> {
    let literal = literal_value(&literal_expr.kind)?;
    match expr_value_ref(value_expr, state) {
        Some(value) => Some(value.map(|value| literal.eq_value(value))),
        None => literal_value(&value_expr.kind).map(|value| Ok(literal == value)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiteralValue {
    Int(i64),
    Bool(bool),
}

impl LiteralValue {
    fn eq_value(self, value: &Value) -> bool {
        matches!(
            (self, value),
            (Self::Int(left), Value::Int(right)) if left == *right
        ) || matches!(
            (self, value),
            (Self::Bool(left), Value::Bool(right)) if left == *right
        )
    }
}

fn literal_value(kind: &CExprKind) -> Option<LiteralValue> {
    match kind {
        CExprKind::Int(value) => Some(LiteralValue::Int(*value)),
        CExprKind::Bool(value) => Some(LiteralValue::Bool(*value)),
        _ => None,
    }
}

fn expr_value_ref<'a>(
    expr: &CExpr,
    state: &'a SlotState,
) -> Option<Result<&'a Value, RuntimeError>> {
    match &expr.kind {
        CExprKind::Var { name, slot } => Some(
            state
                .value(*slot, name)
                .map_err(|error| error.with_span_if_missing(expr.span.clone())),
        ),
        CExprKind::Index {
            target,
            target_slot,
            indices,
        } => Some(
            place_value_parts(target, *target_slot, indices, state)
                .map_err(|error| error.with_span_if_missing(expr.span.clone())),
        ),
        _ => None,
    }
}

fn eval_int(expr: &CExpr, state: &SlotState) -> Result<i64, RuntimeError> {
    match &expr.kind {
        CExprKind::Int(value) => Ok(*value),
        CExprKind::Var { name, slot } => value_as_int(
            state
                .value(*slot, name)
                .map_err(|error| error.with_span_if_missing(expr.span.clone()))?,
            expr.span.clone(),
        ),
        CExprKind::Index {
            target,
            target_slot,
            indices,
        } => value_as_int(
            place_value_parts(target, *target_slot, indices, state)
                .map_err(|error| error.with_span_if_missing(expr.span.clone()))?,
            expr.span.clone(),
        ),
        CExprKind::Unary {
            op: UnaryOp::Neg,
            expr,
        } => Ok(eval_int(expr, state)?.wrapping_neg()),
        CExprKind::Binary { op, left, right } => eval_int_binary(*op, left, right, state, expr),
        _ => value_as_int(&eval(expr, state)?, expr.span.clone()),
    }
}

fn eval_int_binary(
    op: BinaryOp,
    left: &CExpr,
    right: &CExpr,
    state: &SlotState,
    expr: &CExpr,
) -> Result<i64, RuntimeError> {
    match op {
        BinaryOp::Add => Ok(eval_int(left, state)?.wrapping_add(eval_int(right, state)?)),
        BinaryOp::Sub => Ok(eval_int(left, state)?.wrapping_sub(eval_int(right, state)?)),
        BinaryOp::Mul => Ok(eval_int(left, state)?.wrapping_mul(eval_int(right, state)?)),
        BinaryOp::FixedMul => Ok(fixed_mul_q31(
            eval_int(left, state)?,
            eval_int(right, state)?,
        )),
        BinaryOp::Div => {
            let divisor = eval_int(right, state)?;
            if divisor == 0 {
                return Err(RuntimeError::at("division by zero", right.span.clone()));
            }
            Ok(eval_int(left, state)?.wrapping_div(divisor))
        }
        BinaryOp::Rem => {
            let divisor = eval_int(right, state)?;
            if divisor == 0 {
                return Err(RuntimeError::at("remainder by zero", right.span.clone()));
            }
            Ok(eval_int(left, state)?.wrapping_rem(divisor))
        }
        BinaryOp::Shl => Ok(eval_int(left, state)?.wrapping_shl(eval_int(right, state)? as u32)),
        BinaryOp::Shr => Ok(eval_int(left, state)?.wrapping_shr(eval_int(right, state)? as u32)),
        BinaryOp::BitAnd => Ok(eval_int(left, state)? & eval_int(right, state)?),
        BinaryOp::BitXor => Ok(eval_int(left, state)? ^ eval_int(right, state)?),
        BinaryOp::BitOr => Ok(eval_int(left, state)? | eval_int(right, state)?),
        BinaryOp::Eq
        | BinaryOp::NotEq
        | BinaryOp::Lt
        | BinaryOp::LtEq
        | BinaryOp::Gt
        | BinaryOp::GtEq
        | BinaryOp::And
        | BinaryOp::Or => value_as_int(&eval(expr, state)?, expr.span.clone()),
    }
}

fn eval_bool(expr: &CExpr, state: &SlotState) -> Result<bool, RuntimeError> {
    match &expr.kind {
        CExprKind::Bool(value) => Ok(*value),
        CExprKind::Int(value) => Ok(*value != 0),
        CExprKind::Var { name, slot } => value_as_bool(
            state
                .value(*slot, name)
                .map_err(|error| error.with_span_if_missing(expr.span.clone()))?,
            expr.span.clone(),
        ),
        CExprKind::Index {
            target,
            target_slot,
            indices,
        } => value_as_bool(
            place_value_parts(target, *target_slot, indices, state)
                .map_err(|error| error.with_span_if_missing(expr.span.clone()))?,
            expr.span.clone(),
        ),
        CExprKind::Unary {
            op: UnaryOp::Not,
            expr,
        } => Ok(!eval_bool(expr, state)?),
        CExprKind::Binary { op, left, right } => match op {
            BinaryOp::Eq => eval_eq(left, right, state),
            BinaryOp::NotEq => Ok(!eval_eq(left, right, state)?),
            BinaryOp::Lt => Ok(eval_int(left, state)? < eval_int(right, state)?),
            BinaryOp::LtEq => Ok(eval_int(left, state)? <= eval_int(right, state)?),
            BinaryOp::Gt => Ok(eval_int(left, state)? > eval_int(right, state)?),
            BinaryOp::GtEq => Ok(eval_int(left, state)? >= eval_int(right, state)?),
            BinaryOp::And => Ok(eval_bool(left, state)? && eval_bool(right, state)?),
            BinaryOp::Or => Ok(eval_bool(left, state)? || eval_bool(right, state)?),
            _ => value_as_bool(&eval(expr, state)?, expr.span.clone()),
        },
        _ => value_as_bool(&eval(expr, state)?, expr.span.clone()),
    }
}

fn expect_place_int(place: &CPlace, state: &SlotState) -> Result<i64, RuntimeError> {
    match place_value(place, state)? {
        Value::Int(value) => Ok(*value),
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

fn update_place_int(
    place: &CPlace,
    state: &mut SlotState,
    op: UpdateOp,
    rhs: i64,
) -> Result<(), RuntimeError> {
    let current = place_int_mut(place, state)?;
    *current = match op {
        UpdateOp::Add => current.wrapping_add(rhs),
        UpdateOp::Sub => current.wrapping_sub(rhs),
        UpdateOp::Xor => *current ^ rhs,
    };
    Ok(())
}

fn update_place_xor(place: &CPlace, state: &mut SlotState, rhs: Value) -> Result<(), RuntimeError> {
    let display = display_place(place);
    match (place_value_mut(place, state)?, rhs) {
        (Value::Int(current), Value::Int(rhs)) => {
            *current ^= rhs;
            Ok(())
        }
        (Value::Bool(current), Value::Bool(rhs)) => {
            *current ^= rhs;
            Ok(())
        }
        (current, rhs) => Err(RuntimeError::new(format!(
            "expected matching int or bool operands for xor update, found {current} and {rhs} at `{display}`"
        ))),
    }
}

fn place_int_mut<'a>(
    place: &CPlace,
    state: &'a mut SlotState,
) -> Result<&'a mut i64, RuntimeError> {
    let display = display_place(place);
    match place_value_mut(place, state)? {
        Value::Int(value) => Ok(value),
        Value::Bool(value) => Err(RuntimeError::new(format!(
            "expected `{display}` to be int, found bool `{value}`"
        ))),
        Value::Array(value) => Err(RuntimeError::new(format!(
            "expected `{display}` to be int, found array with {} element(s)",
            value.len()
        ))),
        Value::Stack(value) => Err(RuntimeError::new(format!(
            "expected `{display}` to be int, found stack with {} element(s)",
            value.len()
        ))),
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

fn value_as_int(value: &Value, span: Range<usize>) -> Result<i64, RuntimeError> {
    match value {
        Value::Int(value) => Ok(*value),
        Value::Bool(value) => Err(RuntimeError::at(
            format!("expected int, found bool `{value}`"),
            span,
        )),
        Value::Array(value) => Err(RuntimeError::at(
            format!("expected int, found array with {} element(s)", value.len()),
            span,
        )),
        Value::Stack(value) => Err(RuntimeError::at(
            format!("expected int, found stack with {} element(s)", value.len()),
            span,
        )),
    }
}

fn value_as_bool(value: &Value, span: Range<usize>) -> Result<bool, RuntimeError> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Int(value) => Ok(*value != 0),
        Value::Array(value) => Err(RuntimeError::at(
            format!("expected bool, found array with {} element(s)", value.len()),
            span,
        )),
        Value::Stack(value) => Err(RuntimeError::at(
            format!("expected bool, found stack with {} element(s)", value.len()),
            span,
        )),
    }
}

fn push_stack(source: &CPlace, stack: &CStack, state: &mut SlotState) -> Result<(), RuntimeError> {
    let value = expect_place_int(source, state)?;
    assign_place(source, state, Value::Int(0))?;
    expect_stack_mut(&stack.name, stack.slot, 0..0, state)?.insert(0, value);
    Ok(())
}

fn pop_stack(target: &CPlace, stack: &CStack, state: &mut SlotState) -> Result<(), RuntimeError> {
    let current = expect_place_int(target, state)?;
    if current != 0 {
        return Err(RuntimeError::new(format!(
            "pop({}, {}) expected target to be zero, found {current}",
            display_place(target),
            stack.name
        )));
    }

    let stack_value = expect_stack_mut(&stack.name, stack.slot, 0..0, state)?;
    let value = if stack_value.is_empty() {
        return Err(RuntimeError::new(format!(
            "pop expected non-empty stack `{}`",
            stack.name
        )));
    } else {
        stack_value.remove(0)
    };
    assign_place(target, state, Value::Int(value))
}

fn expect_stack<'a>(
    name: &str,
    slot: usize,
    span: Range<usize>,
    state: &'a SlotState,
) -> Result<&'a [i64], RuntimeError> {
    match state
        .value(slot, name)
        .map_err(|error| error.with_span_if_missing(span.clone()))?
    {
        Value::Stack(values) => Ok(values),
        other => Err(RuntimeError::at(
            format!("expected `{name}` to be a stack, found {other}"),
            span,
        )),
    }
}

fn expect_stack_mut<'a>(
    name: &str,
    slot: usize,
    span: Range<usize>,
    state: &'a mut SlotState,
) -> Result<&'a mut Vec<i64>, RuntimeError> {
    match state
        .value_mut(slot, name)
        .map_err(|error| error.with_span_if_missing(span.clone()))?
    {
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
    target: &CPlace,
    expr: &CExpr,
    state: &SlotState,
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
    expr: &CExpr,
    state: &SlotState,
    reads: &mut Vec<ResolvedPlace>,
) -> Result<(), RuntimeError> {
    match &expr.kind {
        CExprKind::Int(_) | CExprKind::Bool(_) | CExprKind::Nil | CExprKind::Size { .. } => {}
        CExprKind::Array(elements) => {
            for element in elements {
                collect_read_places(element, state, reads)?;
            }
        }
        CExprKind::Var { name, .. } => reads.push(ResolvedPlace::Whole(name.clone())),
        CExprKind::Index {
            target, indices, ..
        } => {
            for index in indices {
                collect_read_places(index, state, reads)?;
            }
            reads.push(resolve_place_parts(target, indices, state)?);
        }
        CExprKind::Empty { target, .. } | CExprKind::Top { target, .. } => {
            reads.push(ResolvedPlace::Whole(target.clone()));
        }
        CExprKind::Unary { expr, .. } => collect_read_places(expr, state, reads)?,
        CExprKind::Binary { left, right, .. } => {
            collect_read_places(left, state, reads)?;
            collect_read_places(right, state, reads)?;
        }
    }

    Ok(())
}

fn resolve_place(place: &CPlace, state: &SlotState) -> Result<ResolvedPlace, RuntimeError> {
    resolve_place_parts(&place.name, &place.indices, state)
}

fn resolve_place_parts(
    name: &str,
    indices: &[CExpr],
    state: &SlotState,
) -> Result<ResolvedPlace, RuntimeError> {
    if indices.is_empty() {
        return Ok(ResolvedPlace::Whole(name.to_owned()));
    }

    let indices = checked_indices(indices, state)?
        .into_iter()
        .map(|(index, _)| index)
        .collect();
    Ok(ResolvedPlace::Element {
        name: name.to_owned(),
        indices,
    })
}

fn eval_place(place: &CPlace, state: &SlotState) -> Result<Value, RuntimeError> {
    place_value(place, state).cloned()
}

fn swap_places(left: &CPlace, right: &CPlace, state: &mut SlotState) -> Result<(), RuntimeError> {
    if left.name == right.name
        && left.slot == right.slot
        && let ([left_index], [right_index]) = (left.indices.as_slice(), right.indices.as_slice())
    {
        let left_index = checked_index(left_index, state)?;
        let right_index = checked_index(right_index, state)?;
        let root = state.value_mut(left.slot, &left.name)?;
        return swap_array_elements(&left.name, root, left_index, right_index);
    }

    let left_value = eval_place(left, state)?;
    let right_value = eval_place(right, state)?;
    assign_place(left, state, right_value)?;
    assign_place(right, state, left_value)
}

fn place_value<'a>(place: &CPlace, state: &'a SlotState) -> Result<&'a Value, RuntimeError> {
    place_value_parts(&place.name, place.slot, &place.indices, state)
}

fn place_value_mut<'a>(
    place: &CPlace,
    state: &'a mut SlotState,
) -> Result<&'a mut Value, RuntimeError> {
    if place.indices.is_empty() {
        return state.value_mut(place.slot, &place.name);
    }

    if let [index] = place.indices.as_slice() {
        let index_span = index.span.clone();
        let index = checked_index(index, state)?;
        let root = state.value_mut(place.slot, &place.name)?;
        return array_element_mut(&place.name, root, index, index_span);
    }

    let indices = checked_indices(&place.indices, state)?;
    let root = state.value_mut(place.slot, &place.name)?;
    indexed_place_mut(&place.name, root, &indices)
}

fn place_value_parts<'a>(
    name: &str,
    slot: usize,
    indices: &[CExpr],
    state: &'a SlotState,
) -> Result<&'a Value, RuntimeError> {
    if indices.is_empty() {
        return state.value(slot, name);
    }

    if let [index] = indices {
        let index_value = checked_index(index, state)?;
        return array_element(
            name,
            state.value(slot, name)?,
            index_value,
            index.span.clone(),
        );
    }

    let indices = checked_indices(indices, state)?;
    let mut current = state.value(slot, name)?;
    for (index, index_span) in indices {
        current = array_element(name, current, index, index_span)?;
    }

    Ok(current)
}

fn assign_place(place: &CPlace, state: &mut SlotState, value: Value) -> Result<(), RuntimeError> {
    if place.indices.is_empty() {
        state.set(place.slot, value)?;
        return Ok(());
    }

    if let [index] = place.indices.as_slice() {
        let index_span = index.span.clone();
        let index = checked_index(index, state)?;
        let root = state.value_mut(place.slot, &place.name)?;
        return assign_array_element(&place.name, root, index, index_span, value);
    }

    let indices = checked_indices(&place.indices, state)?;
    let root = state.value_mut(place.slot, &place.name)?;
    assign_indexed_place(&place.name, root, &indices, value)
}

fn array_element<'a>(
    name: &str,
    value: &'a Value,
    index: usize,
    index_span: Range<usize>,
) -> Result<&'a Value, RuntimeError> {
    match value {
        Value::Array(values) => values.get(index).ok_or_else(|| {
            RuntimeError::at(
                format!(
                    "array `{name}` index {index} out of bounds for length {}",
                    values.len()
                ),
                index_span,
            )
        }),
        other => Err(RuntimeError::new(format!(
            "expected `{name}` to be an array, found {other}"
        ))),
    }
}

fn array_element_mut<'a>(
    name: &str,
    value: &'a mut Value,
    index: usize,
    index_span: Range<usize>,
) -> Result<&'a mut Value, RuntimeError> {
    match value {
        Value::Array(values) => {
            let len = values.len();
            values.get_mut(index).ok_or_else(|| {
                RuntimeError::at(
                    format!("array `{name}` index {index} out of bounds for length {len}"),
                    index_span,
                )
            })
        }
        other => Err(RuntimeError::new(format!(
            "expected `{name}` to be an array, found {other}"
        ))),
    }
}

fn assign_array_element(
    name: &str,
    value: &mut Value,
    index: usize,
    index_span: Range<usize>,
    replacement: Value,
) -> Result<(), RuntimeError> {
    match value {
        Value::Array(values) => {
            let len = values.len();
            let slot = values.get_mut(index).ok_or_else(|| {
                RuntimeError::at(
                    format!("array `{name}` index {index} out of bounds for length {len}"),
                    index_span,
                )
            })?;
            *slot = replacement;
            Ok(())
        }
        other => Err(RuntimeError::new(format!(
            "expected `{name}` to be an array, found {other}"
        ))),
    }
}

fn swap_array_elements(
    name: &str,
    value: &mut Value,
    left: usize,
    right: usize,
) -> Result<(), RuntimeError> {
    match value {
        Value::Array(values) => {
            let len = values.len();
            if left >= len {
                return Err(RuntimeError::new(format!(
                    "array `{name}` index {left} out of bounds for length {len}"
                )));
            }
            if right >= len {
                return Err(RuntimeError::new(format!(
                    "array `{name}` index {right} out of bounds for length {len}"
                )));
            }
            values.swap(left, right);
            Ok(())
        }
        other => Err(RuntimeError::new(format!(
            "expected `{name}` to be an array, found {other}"
        ))),
    }
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

fn indexed_place_mut<'a>(
    name: &str,
    root: &'a mut Value,
    indices: &[(usize, Range<usize>)],
) -> Result<&'a mut Value, RuntimeError> {
    let Some(((index, index_span), rest)) = indices.split_first() else {
        return Ok(root);
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
            indexed_place_mut(name, slot, rest)
        }
        other => Err(RuntimeError::new(format!(
            "expected `{name}` to be an array, found {other}"
        ))),
    }
}

fn checked_indices(
    indices: &[CExpr],
    state: &SlotState,
) -> Result<Vec<(usize, Range<usize>)>, RuntimeError> {
    indices
        .iter()
        .map(|index| checked_index(index, state).map(|value| (value, index.span.clone())))
        .collect()
}

fn checked_index(index: &CExpr, state: &SlotState) -> Result<usize, RuntimeError> {
    let value = eval_int(index, state)?;
    usize::try_from(value).map_err(|_| {
        RuntimeError::at(
            format!("array index must be non-negative, found {value}"),
            index.span.clone(),
        )
    })
}

fn assert_condition(
    expr: &CExpr,
    state: &SlotState,
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

fn assert_refinement(
    name: &str,
    refinement: Option<&CExpr>,
    state: &SlotState,
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

#[derive(Debug, Clone, Copy)]
struct CIterateBounds {
    start: i64,
    step: i64,
    end: i64,
}

#[derive(Debug, Clone, Copy)]
struct CIterateRuntime<'a> {
    name: &'a str,
    slot: usize,
    start: &'a CExpr,
    step: &'a CExpr,
    end: &'a CExpr,
    body: &'a CStmt,
}

#[derive(Debug, Clone, Copy)]
struct CProcedureRuntime<'a> {
    name: &'a str,
    target: Option<usize>,
    args: &'a [CPlace],
    runtime_arg_check: bool,
    direct_frame: bool,
    direction: Direction,
}

fn eval_iterate_bounds(
    iterate: CIterateRuntime<'_>,
    state: &SlotState,
    span: Range<usize>,
) -> Result<CIterateBounds, RuntimeError> {
    let start = eval_int(iterate.start, state)?;
    let step = eval_int(iterate.step, state)?;
    let end = eval_int(iterate.end, state)?;
    if step == 0 {
        return Err(RuntimeError::at("iterate step cannot be zero", span));
    }
    Ok(CIterateBounds { start, step, end })
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
    slot: usize,
    expected: i64,
    state: &SlotState,
    span: Range<usize>,
) -> Result<(), RuntimeError> {
    let actual = state
        .value(slot, name)
        .map_err(|error| error.with_span_if_missing(span.clone()))?;
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

enum FrozenCallArgs<'a> {
    Borrowed(&'a [CPlace]),
    Owned(Vec<CPlace>),
}

impl<'a> FrozenCallArgs<'a> {
    fn as_slice(&self) -> &[CPlace] {
        match self {
            Self::Borrowed(args) => args,
            Self::Owned(args) => args,
        }
    }
}

fn freeze_call_args<'a>(
    args: &'a [CPlace],
    state: &SlotState,
) -> Result<FrozenCallArgs<'a>, RuntimeError> {
    if args.iter().all(|arg| arg.indices.is_empty()) {
        return Ok(FrozenCallArgs::Borrowed(args));
    }

    args.iter()
        .map(|arg| freeze_call_arg(arg, state))
        .collect::<Result<Vec<_>, _>>()
        .map(FrozenCallArgs::Owned)
}

fn freeze_call_arg(arg: &CPlace, state: &SlotState) -> Result<CPlace, RuntimeError> {
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
        indices.push(CExpr {
            kind: CExprKind::Int(value),
            span: index.span.clone(),
        });
    }

    Ok(CPlace {
        name: arg.name.clone(),
        slot: arg.slot,
        indices,
    })
}

fn ensure_unique_args(name: &str, args: &[CPlace]) -> Result<(), RuntimeError> {
    for (index, arg) in args.iter().enumerate() {
        for other in args.iter().skip(index + 1) {
            if arg.name == other.name && (arg.indices.is_empty() ^ other.indices.is_empty()) {
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

fn same_call_arg(left: &CPlace, right: &CPlace) -> bool {
    left.name == right.name
        && left.indices.len() == right.indices.len()
        && left
            .indices
            .iter()
            .zip(&right.indices)
            .all(
                |(left, right)| match (const_int_value(left), const_int_value(right)) {
                    (Some(left), Some(right)) => left == right,
                    _ => false,
                },
            )
}

fn call_arg_key(arg: &CPlace) -> String {
    if arg.indices.is_empty() {
        return arg.name.clone();
    }

    let suffix = arg
        .indices
        .iter()
        .map(|index| match &index.kind {
            CExprKind::Int(value) => format!("[{value}]"),
            _ => "[...]".to_owned(),
        })
        .collect::<String>();
    format!("{}{suffix}", arg.name)
}

fn display_place(place: &CPlace) -> String {
    if !place.indices.is_empty() {
        format!("{}{}", place.name, "[...]".repeat(place.indices.len()))
    } else {
        place.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reverie_syntax::parse_program;

    fn compile_body(source: &str) -> CStmt {
        let program = parse_program(source).expect("program parses");
        let mut layout = SlotLayout::default();
        collect_stmt_names(&program.body, &mut layout);
        compile_stmt(&program.body, &layout).expect("program compiles")
    }

    fn update_alias_check(statement: &CStmt) -> bool {
        match &statement.kind {
            CStmtKind::Update { alias_check, .. } => *alias_check,
            other => panic!("expected compiled update, found {other:?}"),
        }
    }

    fn call_runtime_arg_check(statement: &CStmt) -> bool {
        match &statement.kind {
            CStmtKind::Call {
                runtime_arg_check, ..
            } => *runtime_arg_check,
            other => panic!("expected compiled call, found {other:?}"),
        }
    }

    #[test]
    fn skips_alias_scan_for_proven_distinct_constant_array_cells() {
        let statement = compile_body("xs[0] += xs[1]");
        assert!(!update_alias_check(&statement));
    }

    #[test]
    fn keeps_alias_scan_for_same_constant_array_cell() {
        let statement = compile_body("xs[1 + 1] += xs[2]");
        assert!(update_alias_check(&statement));
    }

    #[test]
    fn keeps_alias_scan_for_dynamic_same_root_array_reads() {
        let statement = compile_body("xs[0] += xs[i]");
        assert!(update_alias_check(&statement));
    }

    #[test]
    fn skips_alias_scan_for_different_array_roots() {
        let statement = compile_body("xs[i] += ys[j]");
        assert!(!update_alias_check(&statement));
    }

    #[test]
    fn skips_call_arg_runtime_check_for_proven_constant_element_arguments() {
        let statement = compile_body("proc add(x, y) { x += y }\ncall add(xs[0], xs[1])");
        assert!(!call_runtime_arg_check(&statement));
    }

    #[test]
    fn keeps_call_arg_runtime_check_for_dynamic_element_arguments() {
        let statement = compile_body("proc add(x, y) { x += y }\ncall add(xs[i], ys[j])");
        assert!(call_runtime_arg_check(&statement));
    }

    #[test]
    fn rejects_duplicate_constant_element_arguments_without_runtime_check() {
        let program = parse_program("proc add(x, y) { x += y }\ncall add(xs[1 + 1], xs[2])")
            .expect("program parses");
        let mut layout = SlotLayout::default();
        collect_stmt_names(&program.body, &mut layout);
        let error = compile_stmt(&program.body, &layout).expect_err("duplicate args are rejected");
        assert!(
            error
                .to_string()
                .contains("cannot be called with duplicate argument")
        );
    }
}
