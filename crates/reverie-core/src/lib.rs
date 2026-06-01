use reverie_syntax::{
    BinaryOp, Expr, GlobalDecl, Param, Place, Proc, Program, SourceSpan, Spanned, SpannedExpr,
    SpannedStmt, SpannedType, Stmt, TypeExpr, UnaryOp, UnitExpr, UpdateOp,
};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{message}")]
pub struct CoreError {
    message: String,
    span: SourceSpan,
    labels: Vec<CoreLabel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreLabel {
    span: SourceSpan,
    message: String,
}

impl CoreLabel {
    pub fn new(span: SourceSpan, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
        }
    }

    pub fn span(&self) -> SourceSpan {
        self.span.clone()
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl CoreError {
    pub fn new(message: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            message: message.into(),
            span,
            labels: Vec::new(),
        }
    }

    pub fn with_labels(
        message: impl Into<String>,
        span: SourceSpan,
        labels: Vec<CoreLabel>,
    ) -> Self {
        Self {
            message: message.into(),
            span,
            labels,
        }
    }

    pub fn span(&self) -> SourceSpan {
        self.span.clone()
    }

    pub fn labels(&self) -> &[CoreLabel] {
        &self.labels
    }
}

pub fn check_program(program: &Program) -> Result<(), CoreError> {
    check_program_with_types(program, std::iter::empty::<(String, SpannedType)>())
}

pub fn check_program_with_types(
    program: &Program,
    external_types: impl IntoIterator<Item = (String, SpannedType)>,
) -> Result<(), CoreError> {
    check_globals(&program.globals)?;
    let procedures = procedure_map(&program.procedures)?;
    for procedure in &program.procedures {
        check_procedure(procedure, &procedures)?;
    }

    check_stmt(&program.body, &procedures)?;

    let external_types = external_types.into_iter().collect::<Vec<_>>();
    check_external_global_types(&program.globals, &external_types)?;
    let base_env = TypeEnv::from_globals_and_annotations(&program.globals, external_types);
    check_global_units(&program.globals, &base_env)?;
    for procedure in &program.procedures {
        check_procedure_units(procedure, &procedures, &base_env)?;
    }

    let mut env = base_env;
    check_stmt_units(&program.body, &procedures, &mut env, true)
}

pub fn invert_program(program: &Program) -> Program {
    Program::with_globals_and_procedures(
        program.globals.clone(),
        program.procedures.clone(),
        invert_stmt(&program.body),
    )
}

pub fn invert_stmt(statement: &SpannedStmt) -> SpannedStmt {
    let span = statement.span.clone();
    let node = match &statement.node {
        Stmt::Skip => Stmt::Skip,
        Stmt::Assert { condition } => Stmt::Assert {
            condition: condition.clone(),
        },
        Stmt::Seq(statements) => Stmt::Seq(statements.iter().rev().map(invert_stmt).collect()),
        Stmt::Update { target, op, expr } => Stmt::Update {
            target: target.clone(),
            op: invert_update_op(*op),
            expr: expr.clone(),
        },
        Stmt::Swap { left, right } => Stmt::Swap {
            left: left.clone(),
            right: right.clone(),
        },
        Stmt::Push { source, stack } => Stmt::Pop {
            target: source.clone(),
            stack: stack.clone(),
        },
        Stmt::Pop { target, stack } => Stmt::Push {
            source: target.clone(),
            stack: stack.clone(),
        },
        Stmt::If {
            entry,
            then_branch,
            else_branch,
            exit,
        } => Stmt::If {
            entry: exit.clone(),
            then_branch: Box::new(invert_stmt(then_branch)),
            else_branch: Box::new(invert_stmt(else_branch)),
            exit: entry.clone(),
        },
        Stmt::Loop {
            entry,
            body,
            step,
            exit,
        } => Stmt::Loop {
            entry: exit.clone(),
            body: Box::new(invert_stmt(body)),
            step: Box::new(invert_stmt(step)),
            exit: entry.clone(),
        },
        Stmt::Iterate {
            name,
            start,
            step,
            end,
            body,
        } => Stmt::Iterate {
            name: name.clone(),
            start: end.clone(),
            step: negate_expr(step),
            end: start.clone(),
            body: Box::new(invert_stmt(body)),
        },
        Stmt::Call { name, args } => Stmt::Uncall {
            name: name.clone(),
            args: args.clone(),
        },
        Stmt::Uncall { name, args } => Stmt::Call {
            name: name.clone(),
            args: args.clone(),
        },
        Stmt::Read { target } => Stmt::Unread {
            target: target.clone(),
        },
        Stmt::Unread { target } => Stmt::Read {
            target: target.clone(),
        },
        Stmt::Write { source } => Stmt::Unwrite {
            source: source.clone(),
        },
        Stmt::Unwrite { source } => Stmt::Write {
            source: source.clone(),
        },
        Stmt::Show { targets } => Stmt::Show {
            targets: targets.clone(),
        },
        Stmt::Printf { format, args } => Stmt::Printf {
            format: format.clone(),
            args: args.clone(),
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
        } => Stmt::Local {
            name: name.clone(),
            ty: delocal_ty.clone(),
            dims: delocal_dims.clone(),
            refinement: delocal_refinement.clone(),
            init: delocal.clone(),
            body: Box::new(invert_stmt(body)),
            delocal_name: delocal_name.clone(),
            delocal_ty: ty.clone(),
            delocal_dims: dims.clone(),
            delocal_refinement: refinement.clone(),
            delocal: init.clone(),
        },
        Stmt::Declare {
            name,
            ty,
            len,
            dims,
            init,
        } => Stmt::Declare {
            name: name.clone(),
            ty: ty.clone(),
            len: *len,
            dims: dims.clone(),
            init: init.clone(),
        },
    };

    Spanned::new(node, span)
}

pub fn phase() -> &'static str {
    "v1-arrays"
}

type ProcedureMap<'a> = BTreeMap<&'a str, &'a Proc>;

fn procedure_map(procedures: &[Proc]) -> Result<ProcedureMap<'_>, CoreError> {
    let mut map = BTreeMap::new();
    for procedure in procedures {
        if let Some(previous) = map.insert(procedure.name.as_str(), procedure) {
            return Err(CoreError::with_labels(
                format!("duplicate procedure `{}`", procedure.name),
                procedure.span.clone(),
                vec![
                    CoreLabel::new(previous.span.clone(), "first procedure declaration"),
                    CoreLabel::new(procedure.span.clone(), "duplicate procedure declaration"),
                ],
            ));
        }
    }

    Ok(map)
}

fn check_procedure(procedure: &Proc, procedures: &ProcedureMap<'_>) -> Result<(), CoreError> {
    ensure_unique_param_names(
        &procedure.params,
        procedure.span.clone(),
        format!("procedure `{}` has duplicate parameters", procedure.name),
    )?;
    check_stmt(&procedure.body, procedures)
}

fn check_globals(globals: &[GlobalDecl]) -> Result<(), CoreError> {
    let mut seen = BTreeMap::new();
    for global in globals {
        if global.dims.contains(&0) || global.len == 0 {
            return Err(CoreError::new(
                format!("global `{}` must have length at least 1", global.name),
                global.span.clone(),
            ));
        }

        if global.dims.is_empty() && global.ty.as_ref().is_some_and(is_array_type_annotation) {
            return Err(CoreError::new(
                format!(
                    "global `{}` with array type must declare an explicit length",
                    global.name
                ),
                global.span.clone(),
            ));
        }

        if let Some(previous) = seen.insert(&global.name, global.span.clone()) {
            return Err(CoreError::with_labels(
                format!("duplicate global `{}`", global.name),
                global.span.clone(),
                vec![
                    CoreLabel::new(previous, "first global declaration"),
                    CoreLabel::new(global.span.clone(), "duplicate global declaration"),
                ],
            ));
        }
    }

    Ok(())
}

fn check_stmt(statement: &SpannedStmt, procedures: &ProcedureMap<'_>) -> Result<(), CoreError> {
    match &statement.node {
        Stmt::Skip => Ok(()),
        Stmt::Assert { .. } => Ok(()),
        Stmt::Swap { left, right } => {
            for root in [&left.name, &right.name] {
                if place_index_mentions(left, root) || place_index_mentions(right, root) {
                    return Err(CoreError::new(
                        format!(
                            "irreversible swap: an index expression mentions mutated variable `{root}`"
                        ),
                        statement.span.clone(),
                    ));
                }
            }

            Ok(())
        }
        Stmt::Push { source, .. } => {
            if place_index_mentions(source, &source.name) {
                return Err(CoreError::new(
                    format!(
                        "irreversible push: index expression for `{}` mentions `{}`",
                        display_place(source),
                        source.name
                    ),
                    statement.span.clone(),
                ));
            }

            Ok(())
        }
        Stmt::Pop { target, .. } => {
            if place_index_mentions(target, &target.name) {
                return Err(CoreError::new(
                    format!(
                        "irreversible pop: index expression for `{}` mentions `{}`",
                        display_place(target),
                        target.name
                    ),
                    statement.span.clone(),
                ));
            }

            Ok(())
        }
        Stmt::Seq(statements) => {
            for statement in statements {
                check_stmt(statement, procedures)?;
            }

            Ok(())
        }
        Stmt::Update { target, expr, .. } => {
            if place_index_mentions(target, &target.name) {
                return Err(CoreError::new(
                    format!(
                        "irreversible update: index expression for `{}` mentions `{}`",
                        display_place(target),
                        target.name
                    ),
                    statement.span.clone(),
                ));
            }

            if expr_mentions_forbidden_update_dependency(expr, target) {
                return Err(CoreError::new(
                    format!(
                        "irreversible update: `{}` reads the value being changed",
                        display_place(target)
                    ),
                    expr.span.clone(),
                ));
            }

            Ok(())
        }
        Stmt::Call { name, args } | Stmt::Uncall { name, args } => {
            let procedure = procedures.get(name.as_str()).ok_or_else(|| {
                CoreError::new(
                    format!("unknown procedure `{name}`"),
                    statement.span.clone(),
                )
            })?;

            if procedure.params.len() != args.len() {
                return Err(CoreError::new(
                    format!(
                        "procedure `{name}` expects {} argument(s), found {}",
                        procedure.params.len(),
                        args.len()
                    ),
                    statement.span.clone(),
                ));
            }

            check_call_args(name, args, statement.span.clone())
        }
        Stmt::Read { target } | Stmt::Unread { target } => {
            if place_index_mentions(target, &target.name) {
                return Err(CoreError::new(
                    format!(
                        "irreversible read: index expression for `{}` mentions `{}`",
                        display_place(target),
                        target.name
                    ),
                    statement.span.clone(),
                ));
            }

            Ok(())
        }
        Stmt::Write { .. } | Stmt::Unwrite { .. } => Ok(()),
        Stmt::Show { .. } => Ok(()),
        Stmt::Printf { .. } => Ok(()),
        Stmt::Declare { name, init, .. } => {
            if let Some(init) = init
                && expr_mentions(init, name)
            {
                return Err(CoreError::new(
                    format!("declaration initializer for `{name}` cannot mention `{name}`"),
                    init.span.clone(),
                ));
            }

            Ok(())
        }
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            check_stmt(then_branch, procedures)?;
            check_stmt(else_branch, procedures)
        }
        Stmt::Loop { body, step, .. } => {
            check_stmt(body, procedures)?;
            check_stmt(step, procedures)
        }
        Stmt::Iterate { body, .. } => check_stmt(body, procedures),
        Stmt::Local {
            name,
            ty: _,
            dims: _,
            refinement: _,
            init,
            body,
            delocal_name,
            delocal_ty: _,
            delocal_dims: _,
            delocal_refinement: _,
            delocal,
        } => {
            if name != delocal_name {
                return Err(CoreError::new(
                    format!("local `{name}` must be destroyed by matching `delocal {name}`"),
                    statement.span.clone(),
                ));
            }

            if expr_mentions(init, name) {
                return Err(CoreError::new(
                    format!("local initializer for `{name}` cannot mention `{name}`"),
                    init.span.clone(),
                ));
            }

            if expr_mentions(delocal, name) {
                return Err(CoreError::new(
                    format!("delocal assertion for `{name}` cannot mention `{name}`"),
                    delocal.span.clone(),
                ));
            }

            check_stmt(body, procedures)
        }
    }
}

fn check_call_args(name: &str, args: &[Place], span: SourceSpan) -> Result<(), CoreError> {
    for (index, arg) in args.iter().enumerate() {
        for other in args.iter().skip(index + 1) {
            if arg == other {
                return Err(CoreError::new(
                    format!("procedure `{name}` cannot be called with duplicate arguments"),
                    span,
                ));
            }

            if arg.name == other.name && (!arg.is_indexed() || !other.is_indexed()) {
                return Err(CoreError::new(
                    format!(
                        "procedure `{name}` cannot mix whole-place and element arguments rooted at `{}`",
                        arg.name
                    ),
                    span,
                ));
            }

            if constant_places_alias(arg, other) {
                return Err(CoreError::new(
                    format!("procedure `{name}` cannot be called with duplicate element arguments"),
                    span,
                ));
            }

            if arg.name == other.name && !constant_places_proven_distinct(arg, other) {
                return Err(CoreError::new(
                    format!(
                        "procedure `{name}` cannot be called with potentially aliasing element arguments rooted at `{}`",
                        arg.name
                    ),
                    span,
                ));
            }
        }
    }

    let roots = args
        .iter()
        .map(|arg| arg.name.as_str())
        .collect::<BTreeSet<_>>();
    for arg in args {
        for index in &arg.indices {
            for root in &roots {
                if expr_mentions(index, root) {
                    return Err(CoreError::new(
                        format!(
                            "irreversible call: index expression for `{}` mentions mutable argument `{root}`",
                            display_place(arg)
                        ),
                        index.span.clone(),
                    ));
                }
            }
        }
    }

    Ok(())
}

fn ensure_unique_param_names(
    params: &[Param],
    span: SourceSpan,
    message: String,
) -> Result<(), CoreError> {
    let mut seen = BTreeMap::new();
    for param in params {
        if let Some(previous) = seen.insert(&param.name, param.span.clone()) {
            return Err(CoreError::with_labels(
                message,
                span,
                vec![
                    CoreLabel::new(previous, "first parameter declaration"),
                    CoreLabel::new(param.span.clone(), "duplicate parameter declaration"),
                ],
            ));
        }
    }

    Ok(())
}

fn expr_mentions(expr: &SpannedExpr, name: &str) -> bool {
    match &expr.node {
        Expr::Int { .. } | Expr::Bool(_) | Expr::Nil => false,
        Expr::Array(elements) => elements.iter().any(|element| expr_mentions(element, name)),
        Expr::Var(variable) => variable == name,
        Expr::Index { target, indices } => {
            target == name || indices.iter().any(|index| expr_mentions(index, name))
        }
        Expr::Empty { target } | Expr::Top { target } | Expr::Size { target } => target == name,
        Expr::Unary { expr, .. } => expr_mentions(expr, name),
        Expr::Binary { left, right, .. } => expr_mentions(left, name) || expr_mentions(right, name),
    }
}

fn expr_mentions_forbidden_update_dependency(expr: &SpannedExpr, target: &Place) -> bool {
    match &expr.node {
        Expr::Int { .. } | Expr::Bool(_) | Expr::Nil => false,
        Expr::Array(elements) => elements
            .iter()
            .any(|element| expr_mentions_forbidden_update_dependency(element, target)),
        Expr::Var(variable) => variable == &target.name,
        Expr::Index {
            target: read_target,
            indices,
        } => {
            indices
                .iter()
                .any(|index| expr_mentions(index, &target.name))
                || (read_target == &target.name
                    && (!target.is_indexed()
                        || !constant_indices_proven_distinct(&target.indices, indices)))
        }
        Expr::Empty {
            target: read_target,
        }
        | Expr::Top {
            target: read_target,
        } => read_target == &target.name,
        Expr::Size { .. } => false,
        Expr::Unary { expr, .. } => expr_mentions_forbidden_update_dependency(expr, target),
        Expr::Binary { left, right, .. } => {
            expr_mentions_forbidden_update_dependency(left, target)
                || expr_mentions_forbidden_update_dependency(right, target)
        }
    }
}

fn constant_places_alias(left: &Place, right: &Place) -> bool {
    left.name == right.name && constant_indices_alias(&left.indices, &right.indices)
}

fn constant_places_proven_distinct(left: &Place, right: &Place) -> bool {
    left.name == right.name && constant_indices_proven_distinct(&left.indices, &right.indices)
}

fn constant_indices_alias(left: &[SpannedExpr], right: &[SpannedExpr]) -> bool {
    if left.len() != right.len() {
        return false;
    }

    left.iter().zip(right).all(|(left, right)| {
        match (const_int_value(left), const_int_value(right)) {
            (Some(left), Some(right)) => left == right,
            _ => false,
        }
    })
}

fn constant_indices_proven_distinct(left: &[SpannedExpr], right: &[SpannedExpr]) -> bool {
    left.iter().zip(right).any(|(left, right)| {
        match (const_int_value(left), const_int_value(right)) {
            (Some(left), Some(right)) => left != right,
            _ => false,
        }
    })
}

fn place_index_mentions(place: &Place, name: &str) -> bool {
    place.indices.iter().any(|index| expr_mentions(index, name))
}

fn display_place(place: &Place) -> String {
    if place.is_indexed() {
        format!("{}{}", place.name, "[...]".repeat(place.indices.len()))
    } else {
        place.name.clone()
    }
}

fn invert_update_op(op: UpdateOp) -> UpdateOp {
    match op {
        UpdateOp::Add => UpdateOp::Sub,
        UpdateOp::Sub => UpdateOp::Add,
        UpdateOp::Xor => UpdateOp::Xor,
    }
}

fn negate_expr(expr: &SpannedExpr) -> SpannedExpr {
    if let Expr::Unary {
        op: UnaryOp::Neg,
        expr,
    } = &expr.node
    {
        return (**expr).clone();
    }

    Spanned::new(
        Expr::Unary {
            op: UnaryOp::Neg,
            expr: Box::new(expr.clone()),
        },
        expr.span.clone(),
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct TypeEnv {
    vars: BTreeMap<String, TypeInfo>,
    external: BTreeSet<String>,
    shapes: BTreeMap<String, ArrayShape>,
}

impl TypeEnv {
    fn from_globals_and_annotations(
        globals: &[GlobalDecl],
        types: impl IntoIterator<Item = (String, SpannedType)>,
    ) -> Self {
        let mut env = Self::default();
        for global in globals {
            env.insert_with_shape(
                global.name.clone(),
                global_type(global),
                shape_from_declared_dims(&global.dims),
            );
        }
        for (name, ty) in types {
            env.insert_external(name, type_from_annotation(&ty));
        }
        env
    }

    fn get(&self, name: &str) -> TypeInfo {
        self.vars.get(name).cloned().unwrap_or(TypeInfo::Unknown)
    }

    fn insert(&mut self, name: String, info: TypeInfo) -> Option<TypeInfo> {
        self.external.remove(&name);
        self.shapes.remove(&name);
        self.vars.insert(name, info)
    }

    fn insert_with_shape(
        &mut self,
        name: String,
        info: TypeInfo,
        shape: ArrayShape,
    ) -> Option<TypeInfo> {
        self.external.remove(&name);
        if shape.is_empty() {
            self.shapes.remove(&name);
        } else {
            self.shapes.insert(name.clone(), shape);
        }
        self.vars.insert(name, info)
    }

    fn insert_external(&mut self, name: String, info: TypeInfo) -> Option<TypeInfo> {
        self.external.insert(name.clone());
        self.shapes.remove(&name);
        self.vars.insert(name, info)
    }

    fn remove(&mut self, name: &str) {
        self.vars.remove(name);
        self.external.remove(name);
        self.shapes.remove(name);
    }

    fn contains(&self, name: &str) -> bool {
        self.vars.contains_key(name)
    }

    fn is_external(&self, name: &str) -> bool {
        self.external.contains(name)
    }

    fn shape(&self, name: &str) -> Option<&ArrayShape> {
        self.shapes.get(name)
    }
}

type ArrayShape = Vec<Option<usize>>;
type ShapeEnv = BTreeMap<String, ArrayShape>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum TypeInfo {
    Unknown,
    Int(UnitInfo),
    Bool,
    Array(Box<TypeInfo>),
    Stack,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum UnitInfo {
    Unknown,
    Known(UnitDim),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct UnitDim {
    exponents: BTreeMap<String, i32>,
}

impl UnitDim {
    fn dimensionless() -> Self {
        Self::default()
    }

    fn from_unit(unit: &UnitExpr) -> Self {
        let mut dim = Self::dimensionless();
        for factor in &unit.factors {
            *dim.exponents.entry(factor.name.clone()).or_default() += factor.exponent;
        }
        dim.exponents.retain(|_, exponent| *exponent != 0);
        dim
    }

    fn mul(&self, other: &Self) -> Self {
        self.combine(other, 1)
    }

    fn div(&self, other: &Self) -> Self {
        self.combine(other, -1)
    }

    fn combine(&self, other: &Self, sign: i32) -> Self {
        let mut exponents = self.exponents.clone();
        for (name, exponent) in &other.exponents {
            *exponents.entry(name.clone()).or_default() += sign * exponent;
        }
        exponents.retain(|_, exponent| *exponent != 0);
        Self { exponents }
    }

    fn is_dimensionless(&self) -> bool {
        self.exponents.is_empty()
    }
}

impl std::fmt::Display for UnitDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.exponents.is_empty() {
            return write!(f, "1");
        }

        let mut first = true;
        for (name, exponent) in &self.exponents {
            if !first {
                write!(f, "*")?;
            }
            first = false;

            if *exponent == 1 {
                write!(f, "{name}")?;
            } else {
                write!(f, "{name}^{exponent}")?;
            }
        }

        Ok(())
    }
}

fn check_procedure_units(
    procedure: &Proc,
    procedures: &ProcedureMap<'_>,
    base_env: &TypeEnv,
) -> Result<(), CoreError> {
    let mut env = base_env.clone();
    for param in &procedure.params {
        env.insert(param.name.clone(), param_type(param));
    }

    for param in &procedure.params {
        if let Some(refinement) = &param.refinement {
            expect_bool_expr(refinement, &env)?;
        }
    }

    check_stmt_units(&procedure.body, procedures, &mut env, false)
}

fn param_type(param: &Param) -> TypeInfo {
    match &param.ty {
        Some(ty) => type_from_annotation(ty),
        None => TypeInfo::Unknown,
    }
}

fn global_type(global: &GlobalDecl) -> TypeInfo {
    let ty = if let Some(ty) = &global.ty {
        type_from_annotation(ty)
    } else {
        TypeInfo::Int(UnitInfo::Known(UnitDim::dimensionless()))
    };

    type_with_declaration_dims(&global.dims, ty)
}

fn type_with_declaration_dims(dims: &[usize], ty: TypeInfo) -> TypeInfo {
    let Some((_, rest)) = dims.split_first() else {
        return ty;
    };
    let element_ty = match ty {
        TypeInfo::Array(element) => *element,
        other => other,
    };
    TypeInfo::Array(Box::new(type_with_declaration_dims(rest, element_ty)))
}

fn shape_from_declared_dims(dims: &[usize]) -> ArrayShape {
    dims.iter().copied().map(Some).collect()
}

fn shape_from_local_dims_and_init(dims: &[Option<usize>], init: &SpannedExpr) -> ArrayShape {
    let mut shape = dims.to_vec();
    let Some(literal_shape) = array_literal_shape(init) else {
        return shape;
    };

    if shape.is_empty() {
        return literal_shape;
    }
    for (slot, literal_len) in shape.iter_mut().zip(literal_shape.iter()) {
        if slot.is_none() {
            *slot = *literal_len;
        }
    }
    if literal_shape.len() > shape.len() {
        shape.extend_from_slice(&literal_shape[shape.len()..]);
    }

    shape
}

fn array_literal_shape(expr: &SpannedExpr) -> Option<ArrayShape> {
    let Expr::Array(elements) = &expr.node else {
        return None;
    };

    let mut shape = vec![Some(elements.len())];
    if let Some(first) = elements.first()
        && let Some(mut child_shape) = array_literal_shape(first)
    {
        shape.append(&mut child_shape);
    }
    Some(shape)
}

fn check_global_units(globals: &[GlobalDecl], env: &TypeEnv) -> Result<(), CoreError> {
    for global in globals {
        if let Some(init) = &global.init {
            if expr_mentions(init, &global.name) {
                return Err(CoreError::new(
                    format!(
                        "global initializer for `{}` cannot mention `{}`",
                        global.name, global.name
                    ),
                    init.span.clone(),
                ));
            }

            let declared = global_type(global);
            let init_type = expr_type(init, env)?;
            compatible_types_between(
                &declared,
                &init_type,
                global.span.clone(),
                init.span.clone(),
            )?;
            check_declared_initializer_shape(&global.name, &global.dims, init)?;
        }
    }

    Ok(())
}

fn check_external_global_types(
    globals: &[GlobalDecl],
    external_types: &[(String, SpannedType)],
) -> Result<(), CoreError> {
    for global in globals {
        let declared = global_type(global);
        for (name, ty) in external_types {
            if name == &global.name {
                compatible_types_between(
                    &declared,
                    &type_from_annotation(ty),
                    global.span.clone(),
                    ty.span.clone(),
                )?;
            }
        }
    }

    Ok(())
}

fn type_from_annotation(ty: &SpannedType) -> TypeInfo {
    match &ty.node {
        TypeExpr::Int { unit } => TypeInfo::Int(match unit {
            Some(unit) => UnitInfo::Known(UnitDim::from_unit(&unit.node)),
            None => UnitInfo::Known(UnitDim::dimensionless()),
        }),
        TypeExpr::Bool => TypeInfo::Bool,
        TypeExpr::Array { element } => TypeInfo::Array(Box::new(type_from_annotation(element))),
        TypeExpr::Stack => TypeInfo::Stack,
    }
}

fn is_array_type_annotation(ty: &SpannedType) -> bool {
    matches!(ty.node, TypeExpr::Array { .. })
}

fn check_stmt_units(
    statement: &SpannedStmt,
    procedures: &ProcedureMap<'_>,
    env: &mut TypeEnv,
    specialize_calls: bool,
) -> Result<(), CoreError> {
    match &statement.node {
        Stmt::Skip => Ok(()),
        Stmt::Assert { condition } => expect_condition_expr(condition, env),
        Stmt::Seq(statements) => {
            for statement in statements {
                check_stmt_units(statement, procedures, env, specialize_calls)?;
            }
            Ok(())
        }
        Stmt::Update { target, op, expr } => {
            let target_type = place_type(target, env, statement.span.clone())?;
            let expr_type = expr_type(expr, env)?;

            match op {
                UpdateOp::Add | UpdateOp::Sub => {
                    let target_unit = expect_int_type(&target_type, statement.span.clone())?;
                    let expr_unit = expect_int_type(&expr_type, expr.span.clone())?;
                    compatible_units_between(
                        &target_unit,
                        &expr_unit,
                        statement.span.clone(),
                        expr.span.clone(),
                    )?;
                }
                UpdateOp::Xor => {
                    check_xor_update_type(
                        &target_type,
                        &expr_type,
                        statement.span.clone(),
                        expr.span.clone(),
                    )?;
                }
            }

            Ok(())
        }
        Stmt::Swap { left, right } => {
            let left_type = place_type(left, env, statement.span.clone())?;
            let right_type = place_type(right, env, statement.span.clone())?;
            compatible_types_between(
                &left_type,
                &right_type,
                statement.span.clone(),
                statement.span.clone(),
            )?;
            Ok(())
        }
        Stmt::Push { source, stack }
        | Stmt::Pop {
            target: source,
            stack,
        } => {
            let source_type = place_type(source, env, statement.span.clone())?;
            expect_int_type(&source_type, statement.span.clone())?;
            expect_stack_type(&env.get(stack), statement.span.clone(), stack)?;
            Ok(())
        }
        Stmt::If {
            entry,
            then_branch,
            else_branch,
            exit,
        } => {
            expect_condition_expr(entry, env)?;
            let mut then_env = env.clone();
            check_stmt_units(then_branch, procedures, &mut then_env, specialize_calls)?;
            let mut else_env = env.clone();
            check_stmt_units(else_branch, procedures, &mut else_env, specialize_calls)?;
            expect_condition_expr(exit, env)
        }
        Stmt::Loop {
            entry,
            body,
            step,
            exit,
        } => {
            expect_condition_expr(entry, env)?;
            let mut body_env = env.clone();
            check_stmt_units(body, procedures, &mut body_env, specialize_calls)?;
            let mut step_env = env.clone();
            check_stmt_units(step, procedures, &mut step_env, specialize_calls)?;
            expect_condition_expr(exit, env)
        }
        Stmt::Iterate {
            name,
            start,
            step,
            end,
            body,
        } => {
            if env.contains(name) {
                return Err(CoreError::new(
                    format!("iterate variable `{name}` shadows an existing typed variable"),
                    statement.span.clone(),
                ));
            }

            let start_unit = expect_int_expr(start, env)?;
            let step_unit = expect_int_expr(step, env)?;
            let end_unit = expect_int_expr(end, env)?;
            require_dimensionless(&start_unit, start.span.clone())?;
            require_dimensionless(&step_unit, step.span.clone())?;
            require_dimensionless(&end_unit, end.span.clone())?;
            if is_zero_int_constant(step) {
                return Err(CoreError::new(
                    "iterate step cannot be zero",
                    step.span.clone(),
                ));
            }

            env.insert(
                name.clone(),
                TypeInfo::Int(UnitInfo::Known(UnitDim::dimensionless())),
            );
            check_stmt_units(body, procedures, env, specialize_calls)?;
            env.remove(name);
            Ok(())
        }
        Stmt::Call { name, args } | Stmt::Uncall { name, args } => {
            let procedure = procedures.get(name.as_str()).ok_or_else(|| {
                CoreError::new(
                    format!("unknown procedure `{name}`"),
                    statement.span.clone(),
                )
            })?;

            for (param, arg) in procedure.params.iter().zip(args) {
                compatible_types_between(
                    &param_type(param),
                    &place_type(arg, env, statement.span.clone())?,
                    statement.span.clone(),
                    statement.span.clone(),
                )?;
            }
            if specialize_calls {
                check_procedure_literal_bounds_for_args(procedure, args, env, procedures)?;
            }

            Ok(())
        }
        Stmt::Read { target } | Stmt::Unread { target } => {
            place_type(target, env, statement.span.clone())?;
            Ok(())
        }
        Stmt::Write { source } | Stmt::Unwrite { source } => {
            place_type(source, env, statement.span.clone())?;
            Ok(())
        }
        Stmt::Show { .. } => Ok(()),
        Stmt::Printf { format, args } => check_printf(format, args, statement.span.clone(), env),
        Stmt::Declare {
            name,
            ty,
            dims,
            init,
            ..
        } => {
            reject_zero_declared_dims("declaration", name, dims, statement.span.clone())?;
            if dims.is_empty() && is_array_type_annotation(ty) {
                return Err(CoreError::new(
                    format!("declaration `{name}` with array type must declare an explicit length"),
                    statement.span.clone(),
                ));
            }

            let declared = type_from_annotation(ty);
            if env.contains(name) && !env.is_external(name) {
                return Err(CoreError::new(
                    format!("declaration `{name}` shadows an existing typed variable"),
                    statement.span.clone(),
                ));
            }

            if env.is_external(name) {
                compatible_types_between(
                    &declared,
                    &env.get(name),
                    ty.span.clone(),
                    statement.span.clone(),
                )?;
            }
            if let Some(init) = init {
                let init_type = expr_type(init, env)?;
                compatible_types_between(
                    &declared,
                    &init_type,
                    ty.span.clone(),
                    init.span.clone(),
                )?;
                check_declared_initializer_shape(name, dims, init)?;
            }
            env.insert_with_shape(name.clone(), declared, shape_from_declared_dims(dims));
            Ok(())
        }
        Stmt::Local {
            name,
            ty,
            dims,
            refinement,
            init,
            body,
            delocal_name: _,
            delocal_ty,
            delocal_dims,
            delocal_refinement,
            delocal,
        } => {
            reject_zero_local_dims(name, dims, "local", statement.span.clone())?;
            reject_zero_local_dims(name, delocal_dims, "delocal", statement.span.clone())?;
            if env.contains(name) {
                return Err(CoreError::new(
                    format!("local `{name}` shadows an existing typed variable"),
                    statement.span.clone(),
                ));
            }

            let init_type = expr_type(init, env)?;
            let local_type = match ty {
                Some(ty) => {
                    let annotated = type_from_annotation(ty);
                    compatible_types_between(
                        &annotated,
                        &init_type,
                        ty.span.clone(),
                        init.span.clone(),
                    )?;
                    annotated
                }
                None => init_type,
            };
            check_array_suffix_initializer_shape(name, dims, init)?;

            let local_shape = shape_from_local_dims_and_init(dims, init);
            env.insert_with_shape(name.clone(), local_type.clone(), local_shape);
            if let Some(refinement) = refinement {
                expect_bool_expr(refinement, env)?;
            }
            check_stmt_units(body, procedures, env, specialize_calls)?;
            if let Some(refinement) = refinement {
                expect_bool_expr(refinement, env)?;
            }
            if let Some(delocal_ty) = delocal_ty {
                let annotated = type_from_annotation(delocal_ty);
                compatible_types_between(
                    &local_type,
                    &annotated,
                    statement.span.clone(),
                    delocal_ty.span.clone(),
                )?;
            }
            check_compatible_array_suffix_dims(name, dims, delocal_dims, statement.span.clone())?;
            if let Some(delocal_refinement) = delocal_refinement {
                expect_bool_expr(delocal_refinement, env)?;
            }
            let delocal_type = expr_type(delocal, env)?;
            compatible_types_between(
                &local_type,
                &delocal_type,
                statement.span.clone(),
                delocal.span.clone(),
            )?;
            check_array_suffix_initializer_shape(name, dims, delocal)?;
            check_array_suffix_initializer_shape(name, delocal_dims, delocal)?;
            env.remove(name);
            Ok(())
        }
    }
}

fn reject_zero_declared_dims(
    kind: &str,
    name: &str,
    dims: &[usize],
    span: SourceSpan,
) -> Result<(), CoreError> {
    if dims.contains(&0) {
        return Err(CoreError::new(
            format!("{kind} `{name}` must have length at least 1"),
            span,
        ));
    }
    Ok(())
}

fn reject_zero_local_dims(
    name: &str,
    dims: &[Option<usize>],
    kind: &str,
    span: SourceSpan,
) -> Result<(), CoreError> {
    if dims.contains(&Some(0)) {
        return Err(CoreError::new(
            format!("{kind} `{name}` must have length at least 1"),
            span,
        ));
    }
    Ok(())
}

fn check_declared_initializer_shape(
    name: &str,
    dims: &[usize],
    init: &SpannedExpr,
) -> Result<(), CoreError> {
    let Some((expected_len, rest)) = dims.split_first() else {
        return Ok(());
    };

    let Expr::Array(elements) = &init.node else {
        return Ok(());
    };

    if elements.len() != *expected_len {
        return Err(CoreError::new(
            format!(
                "declaration `{name}` expected array length {expected_len}, found {}",
                elements.len()
            ),
            init.span.clone(),
        ));
    }

    for element in elements {
        check_declared_initializer_shape(name, rest, element)?;
    }

    Ok(())
}

fn check_array_suffix_initializer_shape(
    name: &str,
    dims: &[Option<usize>],
    init: &SpannedExpr,
) -> Result<(), CoreError> {
    let Some((expected_len, rest)) = dims.split_first() else {
        return Ok(());
    };

    let Expr::Array(elements) = &init.node else {
        return Ok(());
    };

    if let Some(expected_len) = expected_len
        && elements.len() != *expected_len
    {
        return Err(CoreError::new(
            format!(
                "local `{name}` expected array length {expected_len}, found {}",
                elements.len()
            ),
            init.span.clone(),
        ));
    }

    for element in elements {
        check_array_suffix_initializer_shape(name, rest, element)?;
    }

    Ok(())
}

fn check_compatible_array_suffix_dims(
    name: &str,
    dims: &[Option<usize>],
    delocal_dims: &[Option<usize>],
    span: SourceSpan,
) -> Result<(), CoreError> {
    for (index, (local_dim, delocal_dim)) in dims.iter().zip(delocal_dims).enumerate() {
        if let (Some(local_dim), Some(delocal_dim)) = (local_dim, delocal_dim)
            && local_dim != delocal_dim
        {
            return Err(CoreError::new(
                format!(
                    "local `{name}` array dimension {} has length {local_dim} but delocal declares {delocal_dim}",
                    index + 1
                ),
                span,
            ));
        }
    }

    Ok(())
}

fn expr_type(expr: &SpannedExpr, env: &TypeEnv) -> Result<TypeInfo, CoreError> {
    match &expr.node {
        Expr::Int { unit, .. } => Ok(TypeInfo::Int(match unit {
            Some(unit) => UnitInfo::Known(UnitDim::from_unit(&unit.node)),
            None => UnitInfo::Known(UnitDim::dimensionless()),
        })),
        Expr::Bool(_) => Ok(TypeInfo::Bool),
        Expr::Array(elements) => array_type(elements, env),
        Expr::Nil => Ok(TypeInfo::Stack),
        Expr::Var(name) => Ok(env.get(name)),
        Expr::Index { target, indices } => {
            let place = Place::with_indices(target.clone(), indices.clone());
            place_type(&place, env, expr.span.clone())
        }
        Expr::Empty { target } => {
            expect_stack_type(&env.get(target), expr.span.clone(), target)?;
            Ok(TypeInfo::Bool)
        }
        Expr::Top { target } => {
            expect_stack_type(&env.get(target), expr.span.clone(), target)?;
            Ok(TypeInfo::Int(UnitInfo::Known(UnitDim::dimensionless())))
        }
        Expr::Size { target } => size_type(target, env, expr.span.clone()),
        Expr::Unary { op, expr } => unary_type(*op, expr, env),
        Expr::Binary { op, left, right } => binary_type(*op, left, right, env),
    }
}

fn array_type(elements: &[SpannedExpr], env: &TypeEnv) -> Result<TypeInfo, CoreError> {
    let Some((first, rest)) = elements.split_first() else {
        return Ok(TypeInfo::Array(Box::new(TypeInfo::Unknown)));
    };

    let element_type = expr_type(first, env)?;
    for element in rest {
        let actual = expr_type(element, env)?;
        compatible_types_between(
            &element_type,
            &actual,
            first.span.clone(),
            element.span.clone(),
        )?;
        check_array_literal_shapes_compatible(first, element)?;
    }

    Ok(TypeInfo::Array(Box::new(element_type)))
}

fn check_array_literal_shapes_compatible(
    expected: &SpannedExpr,
    actual: &SpannedExpr,
) -> Result<(), CoreError> {
    let (Expr::Array(expected_elements), Expr::Array(actual_elements)) =
        (&expected.node, &actual.node)
    else {
        return Ok(());
    };

    if expected_elements.len() != actual_elements.len() {
        return Err(CoreError::new(
            format!(
                "array literal row length mismatch: expected {}, found {}",
                expected_elements.len(),
                actual_elements.len()
            ),
            actual.span.clone(),
        ));
    }

    for (expected_element, actual_element) in expected_elements.iter().zip(actual_elements) {
        check_array_literal_shapes_compatible(expected_element, actual_element)?;
    }

    Ok(())
}

fn place_type(place: &Place, env: &TypeEnv, span: SourceSpan) -> Result<TypeInfo, CoreError> {
    let mut current = env.get(&place.name);
    for (depth, index) in place.indices.iter().enumerate() {
        reject_negative_literal_index(index)?;
        reject_literal_index_out_of_bounds(&place.name, depth, index, env)?;
        let index_unit = expect_int_expr(index, env)?;
        require_dimensionless(&index_unit, index.span.clone())?;

        current = match current {
            TypeInfo::Unknown => TypeInfo::Unknown,
            TypeInfo::Array(element) => *element,
            TypeInfo::Int(_) | TypeInfo::Bool | TypeInfo::Stack => {
                return Err(CoreError::new(
                    format!("expected `{}` to be an array", place.name),
                    span,
                ));
            }
        };
    }

    Ok(current)
}

fn size_type(target: &str, env: &TypeEnv, span: SourceSpan) -> Result<TypeInfo, CoreError> {
    match env.get(target) {
        TypeInfo::Unknown | TypeInfo::Array(_) | TypeInfo::Stack => {
            Ok(TypeInfo::Int(UnitInfo::Known(UnitDim::dimensionless())))
        }
        TypeInfo::Int(_) | TypeInfo::Bool => Err(CoreError::new(
            format!("expected `{target}` to be an array or stack"),
            span,
        )),
    }
}

fn binary_type(
    op: BinaryOp,
    left: &SpannedExpr,
    right: &SpannedExpr,
    env: &TypeEnv,
) -> Result<TypeInfo, CoreError> {
    match op {
        BinaryOp::Add | BinaryOp::Sub => {
            let left_unit = expect_int_expr(left, env)?;
            let right_unit = expect_int_expr(right, env)?;
            let unit = compatible_units_between(
                &left_unit,
                &right_unit,
                left.span.clone(),
                right.span.clone(),
            )?;
            Ok(TypeInfo::Int(unit))
        }
        BinaryOp::Rem => {
            reject_constant_zero_divisor(right, "remainder")?;
            let left_unit = expect_int_expr(left, env)?;
            let right_unit = expect_int_expr(right, env)?;
            let unit = compatible_units_between(
                &left_unit,
                &right_unit,
                left.span.clone(),
                right.span.clone(),
            )?;
            Ok(TypeInfo::Int(unit))
        }
        BinaryOp::Mul => {
            let left_unit = expect_int_expr(left, env)?;
            let right_unit = expect_int_expr(right, env)?;
            Ok(TypeInfo::Int(mul_units(&left_unit, &right_unit)))
        }
        BinaryOp::FixedMul => {
            let left_unit = expect_int_expr(left, env)?;
            let right_unit = expect_int_expr(right, env)?;
            require_dimensionless(&left_unit, left.span.clone())?;
            require_dimensionless(&right_unit, right.span.clone())?;
            Ok(TypeInfo::Int(UnitInfo::Known(UnitDim::dimensionless())))
        }
        BinaryOp::Div => {
            reject_constant_zero_divisor(right, "division")?;
            let left_unit = expect_int_expr(left, env)?;
            let right_unit = expect_int_expr(right, env)?;
            Ok(TypeInfo::Int(div_units(&left_unit, &right_unit)))
        }
        BinaryOp::Shl | BinaryOp::Shr => {
            let left_unit = expect_int_expr(left, env)?;
            let right_unit = expect_int_expr(right, env)?;
            require_dimensionless(&left_unit, left.span.clone())?;
            require_dimensionless(&right_unit, right.span.clone())?;
            Ok(TypeInfo::Int(UnitInfo::Known(UnitDim::dimensionless())))
        }
        BinaryOp::BitAnd | BinaryOp::BitXor | BinaryOp::BitOr => bitwise_type(left, right, env),
        BinaryOp::Eq | BinaryOp::NotEq => {
            let left_type = expr_type(left, env)?;
            let right_type = expr_type(right, env)?;
            compatible_types_between(
                &left_type,
                &right_type,
                left.span.clone(),
                right.span.clone(),
            )?;
            Ok(TypeInfo::Bool)
        }
        BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => {
            let left_unit = expect_int_expr(left, env)?;
            let right_unit = expect_int_expr(right, env)?;
            compatible_units_between(
                &left_unit,
                &right_unit,
                left.span.clone(),
                right.span.clone(),
            )?;
            Ok(TypeInfo::Bool)
        }
        BinaryOp::And | BinaryOp::Or => {
            expect_bool_expr(left, env)?;
            expect_bool_expr(right, env)?;
            Ok(TypeInfo::Bool)
        }
    }
}

fn reject_constant_zero_divisor(expr: &SpannedExpr, operation: &str) -> Result<(), CoreError> {
    if is_zero_int_constant(expr) {
        return Err(CoreError::new(
            format!("{operation} by constant zero"),
            expr.span.clone(),
        ));
    }

    Ok(())
}

fn reject_negative_literal_index(expr: &SpannedExpr) -> Result<(), CoreError> {
    if let Some(value) = const_int_value(expr)
        && value < 0
    {
        return Err(CoreError::new(
            format!("array index must be non-negative, found {value}"),
            expr.span.clone(),
        ));
    }

    Ok(())
}

fn reject_literal_index_out_of_bounds(
    name: &str,
    depth: usize,
    expr: &SpannedExpr,
    env: &TypeEnv,
) -> Result<(), CoreError> {
    reject_literal_index_out_of_bounds_for_shape(name, depth, expr, env.shape(name))
}

fn reject_literal_index_out_of_bounds_for_shape(
    name: &str,
    depth: usize,
    expr: &SpannedExpr,
    shape: Option<&ArrayShape>,
) -> Result<(), CoreError> {
    let Some(value) = const_int_value(expr) else {
        return Ok(());
    };
    let Some(Some(len)) = shape.and_then(|shape| shape.get(depth)) else {
        return Ok(());
    };
    let Ok(len_i64) = i64::try_from(*len) else {
        return Ok(());
    };
    if value >= len_i64 {
        return Err(CoreError::new(
            format!("array `{name}` index {value} out of bounds for length {len}"),
            expr.span.clone(),
        ));
    }

    Ok(())
}

fn check_procedure_literal_bounds_for_args(
    procedure: &Proc,
    args: &[Place],
    caller_env: &TypeEnv,
    procedures: &ProcedureMap<'_>,
) -> Result<(), CoreError> {
    let mut shapes = ShapeEnv::new();
    for (param, arg) in procedure.params.iter().zip(args) {
        let shape = shape_after_place(arg, caller_env);
        if !shape.is_empty() {
            shapes.insert(param.name.clone(), shape);
        }
    }
    let mut visiting = BTreeSet::from([procedure.name.clone()]);

    check_stmt_literal_bounds(&procedure.body, &mut shapes, procedures, &mut visiting)
}

fn shape_after_place(place: &Place, env: &TypeEnv) -> ArrayShape {
    env.shape(&place.name)
        .map(|shape| shape.iter().skip(place.indices.len()).copied().collect())
        .unwrap_or_default()
}

fn shape_after_place_in_shapes(place: &Place, shapes: &ShapeEnv) -> ArrayShape {
    shapes
        .get(&place.name)
        .map(|shape| shape.iter().skip(place.indices.len()).copied().collect())
        .unwrap_or_default()
}

fn check_stmt_literal_bounds(
    statement: &SpannedStmt,
    shapes: &mut ShapeEnv,
    procedures: &ProcedureMap<'_>,
    visiting: &mut BTreeSet<String>,
) -> Result<(), CoreError> {
    match &statement.node {
        Stmt::Skip => Ok(()),
        Stmt::Seq(statements) => {
            for statement in statements {
                check_stmt_literal_bounds(statement, shapes, procedures, visiting)?;
            }
            Ok(())
        }
        Stmt::Assert { condition } => check_expr_literal_bounds(condition, shapes),
        Stmt::Update { target, expr, .. } => {
            check_place_literal_bounds(target, shapes)?;
            check_expr_literal_bounds(expr, shapes)
        }
        Stmt::Swap { left, right } => {
            check_place_literal_bounds(left, shapes)?;
            check_place_literal_bounds(right, shapes)
        }
        Stmt::Push { source, .. }
        | Stmt::Pop { target: source, .. }
        | Stmt::Read { target: source }
        | Stmt::Unread { target: source }
        | Stmt::Write { source }
        | Stmt::Unwrite { source } => check_place_literal_bounds(source, shapes),
        Stmt::If {
            entry,
            then_branch,
            else_branch,
            exit,
        } => {
            check_expr_literal_bounds(entry, shapes)?;
            let mut then_shapes = shapes.clone();
            check_stmt_literal_bounds(then_branch, &mut then_shapes, procedures, visiting)?;
            let mut else_shapes = shapes.clone();
            check_stmt_literal_bounds(else_branch, &mut else_shapes, procedures, visiting)?;
            check_expr_literal_bounds(exit, shapes)
        }
        Stmt::Loop {
            entry,
            body,
            step,
            exit,
        } => {
            check_expr_literal_bounds(entry, shapes)?;
            let mut body_shapes = shapes.clone();
            check_stmt_literal_bounds(body, &mut body_shapes, procedures, visiting)?;
            let mut step_shapes = shapes.clone();
            check_stmt_literal_bounds(step, &mut step_shapes, procedures, visiting)?;
            check_expr_literal_bounds(exit, shapes)
        }
        Stmt::Iterate {
            name,
            start,
            step,
            end,
            body,
        } => {
            check_expr_literal_bounds(start, shapes)?;
            check_expr_literal_bounds(step, shapes)?;
            check_expr_literal_bounds(end, shapes)?;
            shapes.remove(name);
            check_stmt_literal_bounds(body, shapes, procedures, visiting)
        }
        Stmt::Call { name, args } | Stmt::Uncall { name, args } => {
            for arg in args {
                check_place_literal_bounds(arg, shapes)?;
            }
            if visiting.contains(name) {
                return Ok(());
            }
            let Some(procedure) = procedures.get(name.as_str()) else {
                return Ok(());
            };
            let mut callee_shapes = ShapeEnv::new();
            for (param, arg) in procedure.params.iter().zip(args) {
                let shape = shape_after_place_in_shapes(arg, shapes);
                if !shape.is_empty() {
                    callee_shapes.insert(param.name.clone(), shape);
                }
            }
            visiting.insert(name.clone());
            let result = check_stmt_literal_bounds(
                &procedure.body,
                &mut callee_shapes,
                procedures,
                visiting,
            );
            visiting.remove(name);
            result?;
            Ok(())
        }
        Stmt::Show { .. } => Ok(()),
        Stmt::Printf { args, .. } => {
            for arg in args {
                check_expr_literal_bounds(arg, shapes)?;
            }
            Ok(())
        }
        Stmt::Declare {
            name, dims, init, ..
        } => {
            if let Some(init) = init {
                check_expr_literal_bounds(init, shapes)?;
            }
            let shape = shape_from_declared_dims(dims);
            if shape.is_empty() {
                shapes.remove(name);
            } else {
                shapes.insert(name.clone(), shape);
            }
            Ok(())
        }
        Stmt::Local {
            name,
            dims,
            init,
            body,
            delocal,
            ..
        } => {
            check_expr_literal_bounds(init, shapes)?;
            let previous = shapes.insert(name.clone(), shape_from_local_dims_and_init(dims, init));
            check_stmt_literal_bounds(body, shapes, procedures, visiting)?;
            check_expr_literal_bounds(delocal, shapes)?;
            if let Some(previous) = previous {
                shapes.insert(name.clone(), previous);
            } else {
                shapes.remove(name);
            }
            Ok(())
        }
    }
}

fn check_expr_literal_bounds(expr: &SpannedExpr, shapes: &ShapeEnv) -> Result<(), CoreError> {
    match &expr.node {
        Expr::Int { .. } | Expr::Bool(_) | Expr::Nil | Expr::Var(_) => Ok(()),
        Expr::Array(elements) => {
            for element in elements {
                check_expr_literal_bounds(element, shapes)?;
            }
            Ok(())
        }
        Expr::Index { target, indices } => check_indices_literal_bounds(target, indices, shapes),
        Expr::Empty { .. } | Expr::Top { .. } | Expr::Size { .. } => Ok(()),
        Expr::Unary { expr, .. } => check_expr_literal_bounds(expr, shapes),
        Expr::Binary { left, right, .. } => {
            check_expr_literal_bounds(left, shapes)?;
            check_expr_literal_bounds(right, shapes)
        }
    }
}

fn check_place_literal_bounds(place: &Place, shapes: &ShapeEnv) -> Result<(), CoreError> {
    check_indices_literal_bounds(&place.name, &place.indices, shapes)
}

fn check_indices_literal_bounds(
    name: &str,
    indices: &[SpannedExpr],
    shapes: &ShapeEnv,
) -> Result<(), CoreError> {
    for (depth, index) in indices.iter().enumerate() {
        reject_negative_literal_index(index)?;
        reject_literal_index_out_of_bounds_for_shape(name, depth, index, shapes.get(name))?;
        check_expr_literal_bounds(index, shapes)?;
    }
    Ok(())
}

fn is_zero_int_constant(expr: &SpannedExpr) -> bool {
    const_int_value(expr) == Some(0)
}

fn const_int_value(expr: &SpannedExpr) -> Option<i64> {
    match &expr.node {
        Expr::Int { value, .. } => Some(*value),
        Expr::Unary {
            op: UnaryOp::Neg,
            expr,
        } => const_int_value(expr).map(i64::wrapping_neg),
        Expr::Binary { op, left, right } => {
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
        _ => None,
    }
}

fn bitwise_type(
    left: &SpannedExpr,
    right: &SpannedExpr,
    env: &TypeEnv,
) -> Result<TypeInfo, CoreError> {
    let left_type = expr_type(left, env)?;
    let right_type = expr_type(right, env)?;
    match (&left_type, &right_type) {
        (TypeInfo::Bool, TypeInfo::Bool) => Ok(TypeInfo::Bool),
        (TypeInfo::Int(left_unit), TypeInfo::Int(right_unit)) => {
            require_dimensionless(left_unit, left.span.clone())?;
            require_dimensionless(right_unit, right.span.clone())?;
            Ok(TypeInfo::Int(UnitInfo::Known(UnitDim::dimensionless())))
        }
        (TypeInfo::Unknown, _) | (_, TypeInfo::Unknown) => Ok(TypeInfo::Unknown),
        _ => Err(CoreError::with_labels(
            "expected matching int or bool expressions for bitwise operator",
            left.span.start..right.span.end,
            vec![
                CoreLabel::new(left.span.clone(), "left operand"),
                CoreLabel::new(right.span.clone(), "right operand"),
            ],
        )),
    }
}

fn check_xor_update_type(
    target_type: &TypeInfo,
    expr_type: &TypeInfo,
    target_span: SourceSpan,
    expr_span: SourceSpan,
) -> Result<(), CoreError> {
    match (target_type, expr_type) {
        (TypeInfo::Unknown, _) | (_, TypeInfo::Unknown) => Ok(()),
        (TypeInfo::Bool, TypeInfo::Bool) => Ok(()),
        (TypeInfo::Int(target_unit), TypeInfo::Int(expr_unit)) => {
            require_dimensionless(target_unit, target_span)?;
            require_dimensionless(expr_unit, expr_span)
        }
        _ => Err(CoreError::with_labels(
            "expected matching dimensionless int or bool expressions for xor update",
            target_span.start..expr_span.end,
            vec![
                CoreLabel::new(target_span, "target"),
                CoreLabel::new(expr_span, "right-hand side"),
            ],
        )),
    }
}

fn unary_type(op: UnaryOp, expr: &SpannedExpr, env: &TypeEnv) -> Result<TypeInfo, CoreError> {
    match op {
        UnaryOp::Neg => Ok(TypeInfo::Int(expect_int_expr(expr, env)?)),
        UnaryOp::Not => {
            expect_bool_expr(expr, env)?;
            Ok(TypeInfo::Bool)
        }
    }
}

fn expect_int_expr(expr: &SpannedExpr, env: &TypeEnv) -> Result<UnitInfo, CoreError> {
    let info = expr_type(expr, env)?;
    expect_int_type(&info, expr.span.clone())
}

fn expect_int_type(info: &TypeInfo, span: SourceSpan) -> Result<UnitInfo, CoreError> {
    match info {
        TypeInfo::Unknown => Ok(UnitInfo::Unknown),
        TypeInfo::Int(unit) => Ok(unit.clone()),
        TypeInfo::Bool | TypeInfo::Array(_) | TypeInfo::Stack => {
            Err(CoreError::new("expected int expression", span))
        }
    }
}

fn expect_bool_expr(expr: &SpannedExpr, env: &TypeEnv) -> Result<(), CoreError> {
    match expr_type(expr, env)? {
        TypeInfo::Unknown | TypeInfo::Bool => Ok(()),
        TypeInfo::Int(_) | TypeInfo::Array(_) | TypeInfo::Stack => Err(CoreError::new(
            "expected bool expression",
            expr.span.clone(),
        )),
    }
}

fn expect_condition_expr(expr: &SpannedExpr, env: &TypeEnv) -> Result<(), CoreError> {
    match expr_type(expr, env)? {
        TypeInfo::Unknown | TypeInfo::Bool => Ok(()),
        TypeInfo::Int(unit) => require_dimensionless(&unit, expr.span.clone()),
        TypeInfo::Array(_) | TypeInfo::Stack => Err(CoreError::new(
            "expected bool or dimensionless int condition",
            expr.span.clone(),
        )),
    }
}

fn expect_stack_type(info: &TypeInfo, span: SourceSpan, name: &str) -> Result<(), CoreError> {
    match info {
        TypeInfo::Unknown | TypeInfo::Stack => Ok(()),
        TypeInfo::Int(_) | TypeInfo::Bool | TypeInfo::Array(_) => Err(CoreError::new(
            format!("expected `{name}` to be a stack"),
            span,
        )),
    }
}

fn check_printf(
    format: &str,
    args: &[SpannedExpr],
    span: SourceSpan,
    env: &TypeEnv,
) -> Result<(), CoreError> {
    let expected = printf_placeholder_count(format, span.clone())?;
    if expected != args.len() {
        return Err(CoreError::new(
            format!(
                "printf format expects {expected} argument(s), found {}",
                args.len()
            ),
            span,
        ));
    }

    for arg in args {
        let unit = expect_int_expr(arg, env)?;
        require_dimensionless(&unit, arg.span.clone())?;
    }

    Ok(())
}

fn printf_placeholder_count(format: &str, span: SourceSpan) -> Result<usize, CoreError> {
    let mut count = 0;
    let mut chars = format.chars();

    while let Some(ch) = chars.next() {
        if ch != '%' {
            continue;
        }

        let Some(specifier) = chars.next() else {
            return Err(CoreError::new("incomplete printf format specifier", span));
        };

        match specifier {
            '%' => {}
            'd' => count += 1,
            other => {
                return Err(CoreError::new(
                    format!("unsupported printf format specifier `%{other}`"),
                    span,
                ));
            }
        }
    }

    Ok(count)
}

fn compatible_types_between(
    expected: &TypeInfo,
    actual: &TypeInfo,
    expected_span: SourceSpan,
    actual_span: SourceSpan,
) -> Result<(), CoreError> {
    match (expected, actual) {
        (TypeInfo::Unknown, _) | (_, TypeInfo::Unknown) => Ok(()),
        (TypeInfo::Bool, TypeInfo::Bool) => Ok(()),
        (TypeInfo::Stack, TypeInfo::Stack) => Ok(()),
        (TypeInfo::Int(left), TypeInfo::Int(right)) => {
            compatible_units_between(left, right, expected_span, actual_span).map(|_| ())
        }
        (TypeInfo::Array(left), TypeInfo::Array(right)) => {
            compatible_types_between(left, right, expected_span, actual_span)
        }
        (TypeInfo::Bool, TypeInfo::Int(_))
        | (TypeInfo::Int(_), TypeInfo::Bool)
        | (TypeInfo::Array(_), TypeInfo::Bool)
        | (TypeInfo::Bool, TypeInfo::Array(_))
        | (TypeInfo::Array(_), TypeInfo::Int(_))
        | (TypeInfo::Int(_), TypeInfo::Array(_))
        | (TypeInfo::Stack, TypeInfo::Bool)
        | (TypeInfo::Bool, TypeInfo::Stack)
        | (TypeInfo::Stack, TypeInfo::Int(_))
        | (TypeInfo::Int(_), TypeInfo::Stack)
        | (TypeInfo::Stack, TypeInfo::Array(_))
        | (TypeInfo::Array(_), TypeInfo::Stack) => Err(comparison_error(
            "type mismatch",
            "expected",
            display_type_info(expected),
            expected_span,
            "found",
            display_type_info(actual),
            actual_span,
        )),
    }
}

fn compatible_units_between(
    expected: &UnitInfo,
    actual: &UnitInfo,
    expected_span: SourceSpan,
    actual_span: SourceSpan,
) -> Result<UnitInfo, CoreError> {
    match (expected, actual) {
        (UnitInfo::Unknown, UnitInfo::Unknown) => Ok(UnitInfo::Unknown),
        (UnitInfo::Known(unit), UnitInfo::Unknown) | (UnitInfo::Unknown, UnitInfo::Known(unit)) => {
            Ok(UnitInfo::Known(unit.clone()))
        }
        (UnitInfo::Known(expected), UnitInfo::Known(actual)) if expected == actual => {
            Ok(UnitInfo::Known(expected.clone()))
        }
        (UnitInfo::Known(expected), UnitInfo::Known(actual)) => Err(comparison_error(
            format!("unit mismatch: expected `{expected}`, found `{actual}`"),
            "expected unit",
            expected.to_string(),
            expected_span,
            "found unit",
            actual.to_string(),
            actual_span,
        )),
    }
}

fn comparison_error(
    message: impl Into<String>,
    expected_label: &'static str,
    expected: String,
    expected_span: SourceSpan,
    actual_label: &'static str,
    actual: String,
    actual_span: SourceSpan,
) -> CoreError {
    let primary_span = actual_span.clone();
    let mut labels = vec![CoreLabel::new(
        expected_span.clone(),
        format!("{expected_label} `{expected}`"),
    )];

    if expected_span == actual_span {
        labels.push(CoreLabel::new(
            actual_span,
            format!("{actual_label} `{actual}` here"),
        ));
    } else {
        labels.push(CoreLabel::new(
            actual_span,
            format!("{actual_label} `{actual}`"),
        ));
    }

    CoreError::with_labels(message, primary_span, labels)
}

fn display_type_info(info: &TypeInfo) -> String {
    match info {
        TypeInfo::Unknown => "unknown".to_owned(),
        TypeInfo::Bool => "bool".to_owned(),
        TypeInfo::Int(UnitInfo::Unknown) => "int<?>".to_owned(),
        TypeInfo::Int(UnitInfo::Known(unit)) if unit.is_dimensionless() => "int".to_owned(),
        TypeInfo::Int(UnitInfo::Known(unit)) => format!("int<{unit}>"),
        TypeInfo::Array(element) => format!("array<{}>", display_type_info(element)),
        TypeInfo::Stack => "stack".to_owned(),
    }
}

fn mul_units(left: &UnitInfo, right: &UnitInfo) -> UnitInfo {
    match (left, right) {
        (UnitInfo::Known(left), UnitInfo::Known(right)) => UnitInfo::Known(left.mul(right)),
        _ => UnitInfo::Unknown,
    }
}

fn div_units(left: &UnitInfo, right: &UnitInfo) -> UnitInfo {
    match (left, right) {
        (UnitInfo::Known(left), UnitInfo::Known(right)) => UnitInfo::Known(left.div(right)),
        _ => UnitInfo::Unknown,
    }
}

fn require_dimensionless(unit: &UnitInfo, span: SourceSpan) -> Result<(), CoreError> {
    match unit {
        UnitInfo::Unknown => Ok(()),
        UnitInfo::Known(unit) if unit.is_dimensionless() => Ok(()),
        UnitInfo::Known(unit) => Err(CoreError::new(
            format!("expected dimensionless int, found `{unit}`"),
            span,
        )),
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use reverie_syntax::{BinaryOp, Spanned, UpdateOp};
    use reverie_syntax::{parse_program, parse_type_expr};

    use super::*;

    #[test]
    fn accepts_phase_one_program() {
        let program = parse_program("x += 1; x <=> y").expect("program parses");
        check_program(&program).expect("program checks");
    }

    #[test]
    fn rejects_update_that_mentions_its_target() {
        let program = parse_program("x += x + 1").expect("program parses");
        let error = check_program(&program).expect_err("program is irreversible");

        assert!(error.to_string().contains("reads the value being changed"));
    }

    #[test]
    fn rejects_update_that_reads_same_constant_array_cell() {
        let program = parse_program("xs[0] += xs[0]").expect("program parses");
        let error = check_program(&program).expect_err("program is irreversible");

        assert!(error.to_string().contains("reads the value being changed"));
    }

    #[test]
    fn rejects_update_that_reads_same_computed_constant_array_cell() {
        let program = parse_program("xs[1 + 1] += xs[2]").expect("program parses");
        let error = check_program(&program).expect_err("program is irreversible");

        assert!(error.to_string().contains("reads the value being changed"));
    }

    #[test]
    fn allows_update_that_reads_different_constant_array_cell() {
        let program = parse_program("xs[0] += xs[1]").expect("program parses");

        check_program(&program).expect("different constant cell reads are reversible");
    }

    #[test]
    fn rejects_update_that_reads_potentially_same_dynamic_array_cell() {
        let program = parse_program("xs[0] += xs[i]").expect("program parses");
        let error = check_program(&program).expect_err("dynamic cell aliasing is rejected");

        assert!(error.to_string().contains("reads the value being changed"));
    }

    #[test]
    fn inverts_primitive_statements() {
        let program = parse_program("x += 1; y -= 2; z ^= 3; x <=> y").expect("program parses");
        let inverse = invert_program(&program);
        let double_inverse = invert_program(&inverse);

        assert_eq!(double_inverse, program);
    }

    #[test]
    fn assert_inverts_to_itself() {
        let program = parse_program("assert x == 1").expect("program parses");
        let inverse = invert_program(&program);
        let double_inverse = invert_program(&inverse);

        assert_eq!(inverse, program);
        assert_eq!(double_inverse, program);
    }

    #[test]
    fn inverts_conditionals_by_swapping_assertions() {
        let program =
            parse_program("if flag then x += 1 else x -= 1 fi flag").expect("program parses");
        let inverse = invert_program(&program);
        let Stmt::If {
            then_branch,
            else_branch,
            ..
        } = inverse.body.node
        else {
            panic!("expected if");
        };

        assert!(matches!(
            then_branch.node,
            Stmt::Update {
                op: reverie_syntax::UpdateOp::Sub,
                ..
            }
        ));
        assert!(matches!(
            else_branch.node,
            Stmt::Update {
                op: reverie_syntax::UpdateOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn inverts_loops_by_swapping_entry_and_exit_assertions() {
        let program =
            parse_program("from i == 0 do i += 1 loop skip until i == n").expect("program parses");
        let inverse = invert_program(&program);
        let double_inverse = invert_program(&inverse);

        assert_eq!(double_inverse, program);
    }

    #[test]
    fn inverts_iterate_by_reversing_bounds_and_body() {
        let program = parse_program("iterate int i = 0 to 3 x += i end").expect("program parses");
        let inverse = invert_program(&program);
        let Stmt::Iterate {
            start, step, end, ..
        } = &inverse.body.node
        else {
            panic!("expected iterate");
        };

        assert!(matches!(start.node, Expr::Int { value: 3, .. }));
        assert!(matches!(end.node, Expr::Int { value: 0, .. }));
        assert!(matches!(
            step.node,
            Expr::Unary {
                op: UnaryOp::Neg,
                ..
            }
        ));
        assert_eq!(invert_program(&inverse), program);
        check_program(&program).expect("program checks");
    }

    #[test]
    fn rejects_zero_literal_iterate_step() {
        let program =
            parse_program("iterate int i = 0 by 0 to 3 x += i end").expect("program parses");
        let error = check_program(&program).expect_err("zero iterate step is rejected");

        assert!(error.to_string().contains("iterate step cannot be zero"));
    }

    #[test]
    fn rejects_constant_zero_iterate_step() {
        let program =
            parse_program("iterate int i = 0 by 1 - 1 to 3 x += i end").expect("program parses");
        let error = check_program(&program).expect_err("constant zero iterate step is rejected");

        assert!(error.to_string().contains("iterate step cannot be zero"));
    }

    #[test]
    fn rejects_constant_zero_divisor() {
        let program = parse_program("x += y / 0").expect("program parses");
        let error = check_program(&program).expect_err("constant division by zero is rejected");

        assert!(error.to_string().contains("division by constant zero"));
    }

    #[test]
    fn rejects_computed_constant_zero_divisor() {
        let program = parse_program("x += y / (1 - 1)").expect("program parses");
        let error = check_program(&program).expect_err("constant division by zero is rejected");

        assert!(error.to_string().contains("division by constant zero"));
    }

    #[test]
    fn rejects_constant_zero_remainder_divisor() {
        let program = parse_program("x += y % (2 * 0)").expect("program parses");
        let error = check_program(&program).expect_err("constant remainder by zero is rejected");

        assert!(error.to_string().contains("remainder by constant zero"));
    }

    #[test]
    fn rejects_negative_constant_array_index() {
        let program =
            parse_program("local xs: array<int> = [1, 2, 3] xs[-1] += 1 delocal xs = [1, 2, 3]")
                .expect("program parses");
        let error = check_program(&program).expect_err("negative constant index is rejected");

        assert!(
            error
                .to_string()
                .contains("array index must be non-negative, found -1")
        );
    }

    #[test]
    fn rejects_computed_negative_constant_array_index() {
        let program =
            parse_program("local xs: array<int> = [1, 2, 3] xs[1 - 2] += 1 delocal xs = [1, 2, 3]")
                .expect("program parses");
        let error = check_program(&program).expect_err("negative constant index is rejected");

        assert!(
            error
                .to_string()
                .contains("array index must be non-negative, found -1")
        );
    }

    #[test]
    fn rejects_constant_array_index_out_of_bounds_when_shape_is_known() {
        let program =
            parse_program("local xs: array<int> = [1, 2, 3] xs[3] += 1 delocal xs = [1, 2, 3]")
                .expect("program parses");
        let error = check_program(&program).expect_err("constant out-of-bounds index is rejected");

        assert!(
            error
                .to_string()
                .contains("array `xs` index 3 out of bounds for length 3")
        );
    }

    #[test]
    fn rejects_computed_constant_array_index_out_of_bounds_when_shape_is_known() {
        let program =
            parse_program("local xs: array<int> = [1, 2, 3] xs[1 + 2] += 1 delocal xs = [1, 2, 3]")
                .expect("program parses");
        let error = check_program(&program).expect_err("constant out-of-bounds index is rejected");

        assert!(
            error
                .to_string()
                .contains("array `xs` index 3 out of bounds for length 3")
        );
    }

    #[test]
    fn rejects_nested_literal_array_index_out_of_bounds_when_shape_is_known() {
        let program = parse_program(
            "local m: array<array<int>> = [[1, 2], [3, 4]] m[0][2] += 1 delocal m = [[1, 2], [3, 4]]",
        )
        .expect("program parses");
        let error =
            check_program(&program).expect_err("nested constant out-of-bounds index is rejected");

        assert!(
            error
                .to_string()
                .contains("array `m` index 2 out of bounds for length 2")
        );
    }

    #[test]
    fn allows_literal_array_index_without_known_shape() {
        let program = parse_program("xs[3] += 1").expect("program parses");

        check_program(&program).expect("unshaped external array index remains a runtime check");
    }

    #[test]
    fn rejects_procedure_literal_index_out_of_bounds_for_known_argument_shape() {
        let program = parse_program(
            "proc touch(xs: array<int>) { xs[3] += 1 } local xs: array<int> = [1, 2, 3] call touch(xs) delocal xs = [1, 2, 3]",
        )
        .expect("program parses");
        let error = check_program(&program)
            .expect_err("procedure constant out-of-bounds index is rejected at call site");

        assert!(
            error
                .to_string()
                .contains("array `xs` index 3 out of bounds for length 3")
        );
    }

    #[test]
    fn allows_procedure_literal_index_without_known_argument_shape() {
        let program = parse_program("proc touch(xs: array<int>) { xs[3] += 1 } call touch(xs)")
            .expect("program parses");

        check_program(&program).expect("unshaped procedure argument remains a runtime check");
    }

    #[test]
    fn rejects_nested_procedure_literal_index_out_of_bounds_for_known_argument_shape() {
        let program = parse_program(
            "proc inner(xs: array<int>) { xs[3] += 1 } \
             proc outer(xs: array<int>) { call inner(xs) } \
             local xs: array<int> = [1, 2, 3] call outer(xs) delocal xs = [1, 2, 3]",
        )
        .expect("program parses");
        let error = check_program(&program)
            .expect_err("nested procedure constant out-of-bounds index is rejected at call site");

        assert!(
            error
                .to_string()
                .contains("array `xs` index 3 out of bounds for length 3")
        );
    }

    #[test]
    fn checks_and_inverts_procedure_calls_and_locals() {
        let program = parse_program("proc bump(n) { n += 1 } call bump(n)").expect("parses");
        check_program(&program).expect("program checks");

        let inverse = invert_program(&program);
        let Stmt::Uncall { name, args } = &inverse.body.node else {
            panic!("expected call to invert into uncall");
        };
        assert_eq!(name, "bump");
        assert_eq!(args, &[Place::new("n".to_owned(), None)]);
        assert_eq!(invert_program(&inverse), program);
    }

    #[test]
    fn duplicate_procedures_report_both_declarations() {
        let program = parse_program("proc f() { skip } proc f() { skip }").expect("parses");
        let error = check_program(&program).expect_err("duplicate procedure is rejected");

        assert!(error.to_string().contains("duplicate procedure `f`"));
        assert!(
            error
                .labels()
                .iter()
                .any(|label| label.message() == "first procedure declaration")
        );
        assert!(
            error
                .labels()
                .iter()
                .any(|label| label.message() == "duplicate procedure declaration")
        );
    }

    #[test]
    fn duplicate_globals_report_both_declarations() {
        let program = parse_program("global x; global x; skip").expect("parses");
        let error = check_program(&program).expect_err("duplicate global is rejected");

        assert!(error.to_string().contains("duplicate global `x`"));
        assert!(
            error
                .labels()
                .iter()
                .any(|label| label.message() == "first global declaration")
        );
        assert!(
            error
                .labels()
                .iter()
                .any(|label| label.message() == "duplicate global declaration")
        );
    }

    #[test]
    fn duplicate_parameters_report_both_declarations() {
        let program = parse_program("proc f(x, x) { skip } call f(a, b)").expect("parses");
        let error = check_program(&program).expect_err("duplicate parameter is rejected");

        assert!(
            error
                .to_string()
                .contains("procedure `f` has duplicate parameters")
        );
        assert!(
            error
                .labels()
                .iter()
                .any(|label| label.message() == "first parameter declaration")
        );
        assert!(
            error
                .labels()
                .iter()
                .any(|label| label.message() == "duplicate parameter declaration")
        );
    }

    #[test]
    fn formatted_inverse_program_parses_and_checks() {
        let program = parse_program("x += 1; y <=> z").expect("parses");
        let inverse = invert_program(&program);
        let source = reverie_syntax::format_program(&inverse);
        let reparsed = parse_program(&source).expect("inverse source parses");

        check_program(&reparsed).expect("inverse source checks");
    }

    #[test]
    fn rejects_mismatched_local_delocal_names() {
        let program = parse_program("local t = 0 skip delocal u = 0").expect("parses");
        let error = check_program(&program).expect_err("program is invalid");

        assert!(error.to_string().contains("matching `delocal t`"));
    }

    #[test]
    fn rejects_unknown_procedure_calls() {
        let program = parse_program("call missing(n)").expect("parses");
        let error = check_program(&program).expect_err("program is invalid");

        assert!(error.to_string().contains("unknown procedure `missing`"));
    }

    #[test]
    fn rejects_duplicate_call_arguments() {
        let program =
            parse_program("proc swapish(x, y) { x <=> y } call swapish(n, n)").expect("parses");
        let error = check_program(&program).expect_err("program is invalid");

        assert!(error.to_string().contains("duplicate arguments"));
    }

    #[test]
    fn rejects_constant_duplicate_element_call_arguments() {
        let program =
            parse_program("proc add(x, y) { x += y } call add(xs[1 + 1], xs[2])").expect("parses");
        let error = check_program(&program).expect_err("program is invalid");

        assert!(error.to_string().contains("duplicate element arguments"));
    }

    #[test]
    fn accepts_different_constant_element_call_arguments() {
        let program =
            parse_program("proc add(x, y) { x += y } call add(xs[0], xs[1])").expect("parses");

        check_program(&program).expect("different constant element arguments are allowed");
    }

    #[test]
    fn rejects_dynamic_element_call_aliases() {
        let program =
            parse_program("proc add(x, y) { x += y } call add(xs[0], xs[i])").expect("parses");
        let error = check_program(&program).expect_err("dynamic element aliases are rejected");

        assert!(
            error
                .to_string()
                .contains("potentially aliasing element arguments")
        );
    }

    #[test]
    fn accepts_array_element_procedure_arguments() {
        let program = parse_program(
            r#"
proc bump(x) { x += 1 }
local xs: array<int> = [41]
  call bump(xs[0])
delocal xs = [42]
"#,
        )
        .expect("parses");

        check_program(&program).expect("program checks");
    }

    #[test]
    fn rejects_call_argument_index_that_mentions_mutable_argument() {
        let program =
            parse_program("proc bump(x, i) { x += 1 } call bump(xs[i], i)").expect("parses");
        let error = check_program(&program).expect_err("program is invalid");

        assert!(error.to_string().contains("irreversible call"));
    }

    #[test]
    fn accepts_matching_units() {
        let program = parse_program(
            "local distance: int<m> = 100<m> local time: int<s> = 10<s> local speed: int<m/s> = 0<m/s> speed += distance / time delocal speed = 10<m/s> delocal time = 10<s> delocal distance = 100<m>",
        )
        .expect("parses");
        check_program(&program).expect("program checks");
    }

    #[test]
    fn rejects_unit_mismatch() {
        let program =
            parse_program("local speed: int<m/s> = 0<m/s> speed += 1<m> delocal speed = 1<m/s>")
                .expect("parses");
        let error = check_program(&program).expect_err("program has a unit mismatch");

        assert!(error.to_string().contains("unit mismatch"));
        assert!(error.to_string().contains("m*s^-1"));
        assert_eq!(error.labels().len(), 2);
        assert_eq!(error.labels()[0].message(), "expected unit `m*s^-1`");
        assert_eq!(error.labels()[1].message(), "found unit `m`");
    }

    #[test]
    fn checks_top_level_seed_types() {
        let program = parse_program("speed += distance / time").expect("parses");
        let types = [
            (
                "speed".to_owned(),
                parse_type_expr("int<m/s>").expect("type parses"),
            ),
            (
                "distance".to_owned(),
                parse_type_expr("int<m>").expect("type parses"),
            ),
            (
                "time".to_owned(),
                parse_type_expr("int<s>").expect("type parses"),
            ),
        ];

        check_program_with_types(&program, types).expect("program checks with typed seeds");
    }

    #[test]
    fn source_declarations_accept_compatible_external_seed_types() {
        let program = parse_program("int x\nx += 1").expect("parses");
        let types = [("x".to_owned(), parse_type_expr("int").expect("type parses"))];

        check_program_with_types(&program, types)
            .expect("source declaration can assert a compatible seeded variable");
    }

    #[test]
    fn source_declarations_reject_incompatible_external_seed_types() {
        let program = parse_program("int x\nx += 1").expect("parses");
        let types = [(
            "x".to_owned(),
            parse_type_expr("bool").expect("type parses"),
        )];
        let error =
            check_program_with_types(&program, types).expect_err("seed type conflicts with source");

        assert!(error.to_string().contains("type mismatch"));
    }

    #[test]
    fn globals_reject_incompatible_external_seed_types() {
        let program = parse_program("global x; skip").expect("parses");
        let types = [(
            "x".to_owned(),
            parse_type_expr("bool").expect("type parses"),
        )];
        let error =
            check_program_with_types(&program, types).expect_err("seed type conflicts with global");

        assert!(error.to_string().contains("type mismatch"));
    }

    #[test]
    fn typed_global_arrays_check_as_arrays() {
        let program = parse_program("global bool flags[3]; flags[0] ^= true").expect("parses");

        check_program(&program).expect("typed global array checks as an array");
    }

    #[test]
    fn typed_global_array_initializers_check_against_element_type() {
        let program =
            parse_program("global flags[3]: bool = [true, false, true]; skip").expect("parses");

        check_program(&program).expect("typed global array initializer checks");
    }

    #[test]
    fn typed_global_array_initializers_reject_wrong_element_type() {
        let program =
            parse_program("global flags[3]: bool = [true, 0, false]; skip").expect("parses");
        let error = check_program(&program).expect_err("initializer has a mixed element type");

        assert!(error.to_string().contains("type mismatch"));
    }

    #[test]
    fn global_array_initializers_reject_wrong_literal_shape() {
        let program = parse_program("global xs[3] = [1, 2]; skip").expect("parses");
        let error = check_program(&program).expect_err("initializer has wrong shape");

        assert!(error.to_string().contains("expected array length 3"));
    }

    #[test]
    fn global_array_type_annotations_require_explicit_length() {
        let program = parse_program("global xs: array<int> = [1, 2]; skip").expect("parses");
        let error = check_program(&program).expect_err("array global lacks fixed length");

        assert!(
            error
                .to_string()
                .contains("global `xs` with array type must declare an explicit length")
        );
    }

    #[test]
    fn source_declaration_array_initializers_reject_wrong_literal_shape() {
        let program = parse_program("int xs[3][2] = [[1, 2], [3, 4]]").expect("parses");
        let error = check_program(&program).expect_err("initializer has wrong nested shape");

        assert!(error.to_string().contains("expected array length 3"));
    }

    #[test]
    fn source_declaration_array_type_annotations_require_explicit_length() {
        let program = parse_program("array<int> xs = [1, 2]").expect("parses");
        let error = check_program(&program).expect_err("array declaration lacks fixed length");

        assert!(
            error
                .to_string()
                .contains("declaration `xs` with array type must declare an explicit length")
        );
    }

    #[test]
    fn source_declarations_reject_zero_length_arrays() {
        let program = parse_program("int xs[0]").expect("parses");
        let error = check_program(&program).expect_err("zero-length source declaration");

        assert!(
            error
                .to_string()
                .contains("declaration `xs` must have length at least 1")
        );
    }

    #[test]
    fn rejects_top_level_seed_unit_mismatch() {
        let program = parse_program("speed += distance").expect("parses");
        let types = [
            (
                "speed".to_owned(),
                parse_type_expr("int<m/s>").expect("type parses"),
            ),
            (
                "distance".to_owned(),
                parse_type_expr("int<m>").expect("type parses"),
            ),
        ];
        let error = check_program_with_types(&program, types).expect_err("unit mismatch");

        assert!(error.to_string().contains("unit mismatch"));
        assert_eq!(error.labels().len(), 2);
    }

    #[test]
    fn rejects_type_mismatch_with_comparison_labels() {
        let program = parse_program("local flag: int = true skip delocal flag = true")
            .expect("program parses");
        let error = check_program(&program).expect_err("program has a type mismatch");

        assert_eq!(error.to_string(), "type mismatch");
        assert_eq!(error.labels().len(), 2);
        assert_eq!(error.labels()[0].message(), "expected `int`");
        assert_eq!(error.labels()[1].message(), "found `bool`");
    }

    #[test]
    fn rejects_dimensioned_xor() {
        let program =
            parse_program("local x: int<m> = 1<m> x ^= 1<m> delocal x = 1<m>").expect("parses");
        let error = check_program(&program).expect_err("program has a unit mismatch");

        assert!(error.to_string().contains("expected dimensionless int"));
    }

    #[test]
    fn accepts_boolean_xor_update() {
        let program = parse_program("local flag: bool = false flag ^= true delocal flag = true")
            .expect("parses");

        check_program(&program).expect("boolean xor update checks");
    }

    #[test]
    fn rejects_mismatched_boolean_xor_update() {
        let program = parse_program("local flag: bool = false flag ^= 1 delocal flag = false")
            .expect("parses");
        let error = check_program(&program).expect_err("program has a type mismatch");

        assert!(
            error
                .to_string()
                .contains("expected matching dimensionless int or bool")
        );
    }

    #[test]
    fn accepts_array_element_updates_and_swaps() {
        let program = parse_program(
            "local xs: array<int> = [1, 2, 3] xs[1] += 5; xs[0] <=> xs[2] delocal xs = [3, 7, 1]",
        )
        .expect("parses");

        check_program(&program).expect("array program checks");
        assert_eq!(invert_program(&invert_program(&program)), program);
    }

    #[test]
    fn accepts_sized_type_first_local_arrays() {
        let program =
            parse_program("local int xs[3] = [1, 2, 3] xs[1] += 5 delocal int xs[3] = [1, 7, 3]")
                .expect("parses");

        check_program(&program).expect("sized type-first local array checks");
        assert_eq!(invert_program(&invert_program(&program)), program);
    }

    #[test]
    fn rejects_sized_type_first_local_initializer_wrong_shape() {
        let program = parse_program("local int xs[3] = [1, 2] skip delocal int xs[3] = [1, 2]")
            .expect("parses");
        let error = check_program(&program).expect_err("local initializer has wrong shape");

        assert!(
            error
                .to_string()
                .contains("local `xs` expected array length 3, found 2")
        );
    }

    #[test]
    fn rejects_zero_length_type_first_local_arrays() {
        let program =
            parse_program("local int xs[0] = [] skip delocal int xs[0] = []").expect("parses");
        let error = check_program(&program).expect_err("zero-length local array");

        assert!(
            error
                .to_string()
                .contains("local `xs` must have length at least 1")
        );
    }

    #[test]
    fn rejects_mismatched_sized_type_first_local_delocal_dims() {
        let program =
            parse_program("local int xs[3] = [1, 2, 3] skip delocal int xs[2] = [1, 2, 3]")
                .expect("parses");
        let error = check_program(&program).expect_err("local and delocal dimensions mismatch");

        assert!(
            error
                .to_string()
                .contains("dimension 1 has length 3 but delocal declares 2")
        );
    }

    #[test]
    fn rejects_sized_type_first_local_delocal_wrong_shape() {
        let program = parse_program("local int xs[3] = [1, 2, 3] skip delocal int xs[] = [1, 2]")
            .expect("parses");
        let error = check_program(&program).expect_err("delocal assertion has wrong shape");

        assert!(
            error
                .to_string()
                .contains("local `xs` expected array length 3, found 2")
        );
    }

    #[test]
    fn rejects_mismatched_type_first_delocal_annotation() {
        let program = parse_program("local int x = 0 skip delocal bool x = 0").expect("parses");
        let error = check_program(&program).expect_err("delocal type annotation mismatches local");

        assert!(error.to_string().contains("type mismatch"));
    }

    #[test]
    fn accepts_type_first_delocal_refinements() {
        let program =
            parse_program("local int x = 0 x += 1 delocal int x where x > 0 = 1").expect("parses");

        check_program(&program).expect("program checks");
        assert_eq!(invert_program(&invert_program(&program)), program);
    }

    #[test]
    fn accepts_reverie_style_delocal_refinements() {
        let program = parse_program("local x: int = 0 x += 1 delocal x: int where x > 0 = 1")
            .expect("parses");

        check_program(&program).expect("program checks");
        assert_eq!(invert_program(&invert_program(&program)), program);
    }

    #[test]
    fn accepts_multidimensional_array_elements() {
        let program = parse_program(
            "local xs: array<array<int>> = [[1, 2], [3, 4]] xs[1][0] += 5 delocal xs = [[1, 2], [8, 4]]",
        )
        .expect("parses");

        check_program(&program).expect("matrix element program checks");
        assert_eq!(invert_program(&invert_program(&program)), program);
    }

    #[test]
    fn rejects_ragged_array_literals() {
        let program = parse_program(
            "local matrix: array<array<int>> = [[1, 2], [3]] skip delocal matrix = [[1, 2], [3]]",
        )
        .expect("parses");
        let error = check_program(&program).expect_err("ragged literal is rejected");

        assert!(
            error
                .to_string()
                .contains("array literal row length mismatch: expected 2, found 1")
        );
    }

    #[test]
    fn rejects_nested_ragged_array_literals() {
        let program = parse_program(
            "local cube: array<array<array<int>>> = [[[1]], [[2, 3]]] skip delocal cube = [[[1]], [[2, 3]]]",
        )
        .expect("parses");
        let error = check_program(&program).expect_err("nested ragged literal is rejected");

        assert!(
            error
                .to_string()
                .contains("array literal row length mismatch: expected 1, found 2")
        );
    }

    #[test]
    fn accepts_dimensionless_int_conditions_for_legacy_janus() {
        let program = parse_program("if mask & 4 then x += 1 fi x == 1").expect("parses");

        check_program(&program).expect("int mask condition checks");
    }

    #[test]
    fn rejects_array_update_from_same_root_when_exact_cell_is_not_statically_known() {
        let program = parse_program("xs[i][j] += xs[i][k]").expect("parses");
        let error = check_program(&program).expect_err("dynamic same-root read aliases target");

        assert!(error.to_string().contains("reads the value being changed"));
    }

    #[test]
    fn accepts_array_update_from_proven_distinct_constant_cells() {
        let program = parse_program("xs[i][0] += xs[j][1]").expect("parses");

        check_program(&program).expect("constant distinct nested cells cannot alias");
    }

    #[test]
    fn accepts_unitful_array_annotations() {
        let program =
            parse_program("local xs: array<int<m>> = [1<m>, 2<m>] skip delocal xs = [1<m>, 2<m>]")
                .expect("parses");

        check_program(&program).expect("unitful array program checks");
    }

    #[test]
    fn rejects_update_rhs_index_that_mentions_target_array() {
        let program = parse_program("local xs = [1, 2] xs[0] += ys[xs[0]] delocal xs = [1, 2]")
            .expect("parses");
        let error = check_program(&program).expect_err("program is irreversible");

        assert!(error.to_string().contains("reads the value being changed"));
    }

    #[test]
    fn rejects_array_index_that_mentions_mutated_swap_root() {
        let program = parse_program("xs[i] <=> i").expect("parses");
        let error = check_program(&program).expect_err("program is irreversible");

        assert!(
            error
                .to_string()
                .contains("index expression mentions mutated variable")
        );
    }

    #[test]
    fn accepts_refinements() {
        let program = parse_program("proc f(n: int where n >= 0) { n += 1; n -= 1 } call f(x)")
            .expect("parses");
        check_program(&program).expect("program checks");
    }

    #[test]
    fn rejects_non_bool_refinement() {
        let program =
            parse_program("proc f(n: int where n + 1) { skip } call f(x)").expect("parses");
        let error = check_program(&program).expect_err("program has invalid refinement");

        assert!(error.to_string().contains("expected bool expression"));
    }

    #[test]
    fn rejects_non_bool_logical_not() {
        let program = parse_program("assert !1").expect("parses");
        let error = check_program(&program).expect_err("program has invalid logical not");

        assert!(error.to_string().contains("expected bool expression"));
    }

    #[test]
    fn accepts_unitful_unary_numeric_negation() {
        let program = parse_program(
            "local distance: int<m> = 1<m> assert -distance == 0<m> - distance delocal distance = 1<m>",
        )
        .expect("parses");

        check_program(&program).expect("program checks");
    }

    #[test]
    fn rejects_non_int_unary_numeric_negation() {
        let program = parse_program("assert -true == 0").expect("parses");
        let error = check_program(&program).expect_err("program has invalid numeric negation");

        assert!(error.to_string().contains("expected int expression"));
    }

    #[test]
    fn accepts_array_size_expression() {
        let program = parse_program(
            "local xs: array<int> = [1, 2, 3] assert size(xs) == 3 delocal xs = [1, 2, 3]",
        )
        .expect("parses");

        check_program(&program).expect("program checks");
    }

    #[test]
    fn accepts_stack_operations_and_expressions() {
        let program = parse_program(
            "local s: stack = nil local x = 1 push(x, s) pop(x, s) delocal x = 1 delocal s = nil",
        )
        .expect("parses");

        check_program(&program).expect("stack program checks");
        assert_eq!(invert_program(&invert_program(&program)), program);
    }

    #[test]
    fn rejects_stack_operations_on_non_stack_values() {
        let program = parse_program("local s: int = 0 push(x, s) delocal s = 0").expect("parses");
        let error = check_program(&program).expect_err("s is not a stack");

        assert!(error.to_string().contains("expected `s` to be a stack"));
    }

    #[test]
    fn rejects_printf_argument_count_mismatch() {
        let program = parse_program(r#"printf("%d %d", x)"#).expect("program parses");
        let error = check_program(&program).expect_err("program has a bad printf");

        assert!(error.to_string().contains("expects 2 argument(s), found 1"));
    }

    #[test]
    fn rejects_printf_unsupported_format_specifier() {
        let program = parse_program(r#"printf("%q", x)"#).expect("program parses");
        let error = check_program(&program).expect_err("program has a bad printf");

        assert!(
            error
                .to_string()
                .contains("unsupported printf format specifier `%q`")
        );
    }

    #[test]
    fn rejects_size_of_non_array() {
        let program =
            parse_program("local n: int = 0 assert size(n) == 0 delocal n = 0").expect("parses");
        let error = check_program(&program).expect_err("size target is not an array");

        assert!(
            error
                .to_string()
                .contains("expected `n` to be an array or stack")
        );
    }

    proptest! {
        #[test]
        fn invert_is_an_involution(program in arbitrary_program()) {
            let inverse = invert_program(&program);
            let double_inverse = invert_program(&inverse);

            prop_assert_eq!(double_inverse, program);
        }
    }

    fn arbitrary_program() -> impl Strategy<Value = Program> {
        arbitrary_stmt().prop_map(Program::new)
    }

    fn arbitrary_stmt() -> impl Strategy<Value = SpannedStmt> {
        let leaf = prop_oneof![
            skip_stmt(),
            assert_stmt(),
            update_stmt(),
            swap_stmt(),
            call_stmt(),
            uncall_stmt()
        ];

        leaf.prop_recursive(4, 32, 3, |inner| {
            let seq = prop::collection::vec(inner.clone(), 1..4)
                .prop_map(|statements| Spanned::new(Stmt::Seq(statements), 0..0));
            let if_stmt = (
                arbitrary_expr(),
                inner.clone(),
                inner.clone(),
                arbitrary_expr(),
            )
                .prop_map(|(entry, then_branch, else_branch, exit)| {
                    Spanned::new(
                        Stmt::If {
                            entry,
                            then_branch: Box::new(then_branch),
                            else_branch: Box::new(else_branch),
                            exit,
                        },
                        0..0,
                    )
                });
            let loop_stmt = (
                arbitrary_expr(),
                inner.clone(),
                inner.clone(),
                arbitrary_expr(),
            )
                .prop_map(|(entry, body, step, exit)| {
                    Spanned::new(
                        Stmt::Loop {
                            entry,
                            body: Box::new(body),
                            step: Box::new(step),
                            exit,
                        },
                        0..0,
                    )
                });
            let local_stmt =
                (arbitrary_expr(), inner, arbitrary_expr()).prop_map(|(init, body, delocal)| {
                    Spanned::new(
                        Stmt::Local {
                            name: "t".to_owned(),
                            ty: None,
                            dims: Vec::new(),
                            refinement: None,
                            init,
                            body: Box::new(body),
                            delocal_name: "t".to_owned(),
                            delocal_ty: None,
                            delocal_dims: Vec::new(),
                            delocal_refinement: None,
                            delocal,
                        },
                        0..0,
                    )
                });

            prop_oneof![seq, if_stmt, loop_stmt, local_stmt]
        })
    }

    fn skip_stmt() -> impl Strategy<Value = SpannedStmt> {
        Just(Spanned::new(Stmt::Skip, 0..0))
    }

    fn assert_stmt() -> impl Strategy<Value = SpannedStmt> {
        arbitrary_expr().prop_map(|condition| Spanned::new(Stmt::Assert { condition }, 0..0))
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

        (targets, ops, arbitrary_expr())
            .prop_map(|(target, op, expr)| Spanned::new(Stmt::Update { target, op, expr }, 0..0))
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

    fn call_stmt() -> impl Strategy<Value = SpannedStmt> {
        Just(Spanned::new(
            Stmt::Call {
                name: "f".to_owned(),
                args: vec![Place::new("x".to_owned(), None)],
            },
            0..0,
        ))
    }

    fn uncall_stmt() -> impl Strategy<Value = SpannedStmt> {
        Just(Spanned::new(
            Stmt::Uncall {
                name: "f".to_owned(),
                args: vec![Place::new("x".to_owned(), None)],
            },
            0..0,
        ))
    }

    fn arbitrary_expr() -> impl Strategy<Value = SpannedExpr> {
        let leaf = prop_oneof![
            any::<i64>().prop_map(|value| { Spanned::new(Expr::Int { value, unit: None }, 0..0,) }),
            any::<bool>().prop_map(|value| Spanned::new(Expr::Bool(value), 0..0)),
            prop_oneof![
                Just("x".to_owned()),
                Just("y".to_owned()),
                Just("z".to_owned())
            ]
            .prop_map(|name| Spanned::new(Expr::Var(name), 0..0)),
            Just(Spanned::new(
                Expr::Size {
                    target: "xs".to_owned(),
                },
                0..0,
            )),
        ];

        leaf.prop_recursive(3, 16, 2, |inner| {
            prop_oneof![
                inner.clone().prop_map(|expr| {
                    Spanned::new(
                        Expr::Unary {
                            op: UnaryOp::Neg,
                            expr: Box::new(expr),
                        },
                        0..0,
                    )
                }),
                inner.clone().prop_map(|expr| {
                    Spanned::new(
                        Expr::Unary {
                            op: UnaryOp::Not,
                            expr: Box::new(expr),
                        },
                        0..0,
                    )
                }),
                (binary_op(), inner.clone(), inner).prop_map(|(op, left, right)| {
                    Spanned::new(
                        Expr::Binary {
                            op,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                        0..0,
                    )
                })
            ]
        })
    }

    fn binary_op() -> BoxedStrategy<BinaryOp> {
        prop_oneof![
            Just(BinaryOp::Add),
            Just(BinaryOp::Sub),
            Just(BinaryOp::Mul),
            Just(BinaryOp::FixedMul),
            Just(BinaryOp::Div),
            Just(BinaryOp::Rem),
            Just(BinaryOp::Shl),
            Just(BinaryOp::Shr),
            Just(BinaryOp::BitAnd),
            Just(BinaryOp::BitXor),
            Just(BinaryOp::BitOr),
            Just(BinaryOp::Eq),
            Just(BinaryOp::NotEq),
            Just(BinaryOp::Lt),
            Just(BinaryOp::LtEq),
            Just(BinaryOp::Gt),
            Just(BinaryOp::GtEq),
            Just(BinaryOp::And),
            Just(BinaryOp::Or),
        ]
        .boxed()
    }
}
