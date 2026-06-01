use crate::{
    BinaryOp, Expr, GlobalDecl, Param, Place, Proc, Program, SpannedExpr, SpannedStmt, SpannedType,
    SpannedUnit, Stmt, TypeExpr, UnaryOp, UnitExpr, UnitFactor, UpdateOp,
};

const INDENT: &str = "  ";

pub fn format_program(program: &Program) -> String {
    let mut out = String::new();

    for global in &program.globals {
        format_global_into(global, &mut out);
        out.push('\n');
    }

    if !program.globals.is_empty() {
        out.push('\n');
    }

    for (index, procedure) in program.procedures.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        format_proc_into(procedure, &mut out);
        out.push('\n');
    }

    if !program.procedures.is_empty() {
        out.push('\n');
    }

    format_stmt_into(&program.body, 0, &mut out);
    out.push('\n');
    out
}

fn format_global_into(global: &GlobalDecl, out: &mut String) {
    out.push_str("global ");
    out.push_str(&global.name);
    for dim in &global.dims {
        out.push('[');
        out.push_str(&dim.to_string());
        out.push(']');
    }
    if let Some(ty) = &global.ty {
        out.push_str(": ");
        out.push_str(&format_spanned_type(ty));
    }
    if let Some(init) = &global.init {
        out.push_str(" = ");
        out.push_str(&format_expr(init));
    }
    out.push(';');
}

pub fn format_stmt(statement: &SpannedStmt) -> String {
    let mut out = String::new();
    format_stmt_into(statement, 0, &mut out);
    out
}

pub fn format_expr(expr: &SpannedExpr) -> String {
    format_expr_with_prec(expr, 0)
}

pub fn format_type_expr(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Int { unit: None } => "int".to_owned(),
        TypeExpr::Int { unit: Some(unit) } if unit.node.factors.is_empty() => "int".to_owned(),
        TypeExpr::Int { unit: Some(unit) } => format!("int<{}>", format_unit_expr(&unit.node)),
        TypeExpr::Bool => "bool".to_owned(),
        TypeExpr::Array { element } => format!("array<{}>", format_spanned_type(element)),
        TypeExpr::Stack => "stack".to_owned(),
    }
}

pub fn format_unit_expr(unit: &UnitExpr) -> String {
    if unit.factors.is_empty() {
        return "1".to_owned();
    }

    let numerator = unit
        .factors
        .iter()
        .filter(|factor| factor.exponent >= 0)
        .collect::<Vec<_>>();
    let denominator = unit
        .factors
        .iter()
        .filter(|factor| factor.exponent < 0)
        .collect::<Vec<_>>();

    let mut out = String::new();

    if numerator.is_empty() {
        let first_denominator = denominator
            .first()
            .expect("denominator exists when unit factors are not empty");
        out.push_str(&format_unit_factor(first_denominator, 0));
    } else {
        out.push_str(
            &numerator
                .iter()
                .map(|factor| format_unit_factor(factor, i64::from(factor.exponent)))
                .collect::<Vec<_>>()
                .join("*"),
        );
    }

    for factor in denominator {
        out.push('/');
        out.push_str(&format_unit_factor(factor, -i64::from(factor.exponent)));
    }

    out
}

fn format_proc_into(procedure: &Proc, out: &mut String) {
    out.push_str("proc ");
    out.push_str(&procedure.name);
    out.push('(');
    out.push_str(
        &procedure
            .params
            .iter()
            .map(format_param)
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str(") {\n");
    format_stmt_into(&procedure.body, 1, out);
    out.push('\n');
    out.push('}');
}

fn format_param(param: &Param) -> String {
    let mut out = param.name.clone();
    format_annotation_into(param.ty.as_ref(), param.refinement.as_ref(), &mut out);
    out
}

fn format_stmt_into(statement: &SpannedStmt, indent: usize, out: &mut String) {
    match &statement.node {
        Stmt::Skip => {
            push_indent(out, indent);
            out.push_str("skip");
        }
        Stmt::Assert { condition } => {
            push_indent(out, indent);
            out.push_str("assert ");
            out.push_str(&format_expr(condition));
        }
        Stmt::Seq(statements) => {
            for (index, statement) in statements.iter().enumerate() {
                if index > 0 {
                    out.push_str(";\n");
                }
                format_stmt_into(statement, indent, out);
            }
        }
        Stmt::Update { target, op, expr } => {
            push_indent(out, indent);
            out.push_str(&format_place(target));
            out.push(' ');
            out.push_str(update_op_text(*op));
            out.push(' ');
            out.push_str(&format_expr(expr));
        }
        Stmt::Swap { left, right } => {
            push_indent(out, indent);
            out.push_str(&format_place(left));
            out.push_str(" <=> ");
            out.push_str(&format_place(right));
        }
        Stmt::Push { source, stack } => {
            push_indent(out, indent);
            out.push_str("push(");
            out.push_str(&format_place(source));
            out.push_str(", ");
            out.push_str(stack);
            out.push(')');
        }
        Stmt::Pop { target, stack } => {
            push_indent(out, indent);
            out.push_str("pop(");
            out.push_str(&format_place(target));
            out.push_str(", ");
            out.push_str(stack);
            out.push(')');
        }
        Stmt::If {
            entry,
            then_branch,
            else_branch,
            exit,
        } => {
            push_indent(out, indent);
            out.push_str("if ");
            out.push_str(&format_expr(entry));
            out.push_str(" then\n");
            format_stmt_into(then_branch, indent + 1, out);
            out.push('\n');
            push_indent(out, indent);
            out.push_str("else\n");
            format_stmt_into(else_branch, indent + 1, out);
            out.push('\n');
            push_indent(out, indent);
            out.push_str("fi ");
            out.push_str(&format_expr(exit));
        }
        Stmt::Loop {
            entry,
            body,
            step,
            exit,
        } => {
            push_indent(out, indent);
            out.push_str("from ");
            out.push_str(&format_expr(entry));
            out.push_str(" do\n");
            format_stmt_into(body, indent + 1, out);
            out.push('\n');
            push_indent(out, indent);
            out.push_str("loop\n");
            format_stmt_into(step, indent + 1, out);
            out.push('\n');
            push_indent(out, indent);
            out.push_str("until ");
            out.push_str(&format_expr(exit));
        }
        Stmt::Iterate {
            name,
            start,
            step,
            end,
            body,
        } => {
            push_indent(out, indent);
            out.push_str("iterate int ");
            out.push_str(name);
            out.push_str(" = ");
            out.push_str(&format_expr(start));
            if !is_default_iterate_step(step) {
                out.push_str(" by ");
                out.push_str(&format_expr(step));
            }
            out.push_str(" to ");
            out.push_str(&format_expr(end));
            out.push('\n');
            format_stmt_into(body, indent + 1, out);
            out.push('\n');
            push_indent(out, indent);
            out.push_str("end");
        }
        Stmt::Call { name, args } => {
            push_indent(out, indent);
            out.push_str("call ");
            out.push_str(name);
            out.push('(');
            out.push_str(&args.iter().map(format_place).collect::<Vec<_>>().join(", "));
            out.push(')');
        }
        Stmt::Uncall { name, args } => {
            push_indent(out, indent);
            out.push_str("uncall ");
            out.push_str(name);
            out.push('(');
            out.push_str(&args.iter().map(format_place).collect::<Vec<_>>().join(", "));
            out.push(')');
        }
        Stmt::Read { target } => {
            push_indent(out, indent);
            out.push_str("read ");
            out.push_str(&format_place(target));
        }
        Stmt::Unread { target } => {
            push_indent(out, indent);
            out.push_str("unread ");
            out.push_str(&format_place(target));
        }
        Stmt::Write { source } => {
            push_indent(out, indent);
            out.push_str("write ");
            out.push_str(&format_place(source));
        }
        Stmt::Unwrite { source } => {
            push_indent(out, indent);
            out.push_str("unwrite ");
            out.push_str(&format_place(source));
        }
        Stmt::Show { targets } => {
            push_indent(out, indent);
            out.push_str("show(");
            out.push_str(&targets.join(", "));
            out.push(')');
        }
        Stmt::Printf { format, args } => {
            push_indent(out, indent);
            out.push_str("printf(\"");
            out.push_str(&escape_string(format));
            out.push('"');
            if !args.is_empty() {
                out.push_str(", ");
                out.push_str(&args.iter().map(format_expr).collect::<Vec<_>>().join(", "));
            }
            out.push(')');
        }
        Stmt::Declare {
            name,
            ty,
            dims,
            init,
            ..
        } => {
            push_indent(out, indent);
            out.push_str(&format_spanned_type(ty));
            out.push(' ');
            out.push_str(name);
            for dim in dims {
                out.push('[');
                out.push_str(&dim.to_string());
                out.push(']');
            }
            if let Some(init) = init {
                out.push_str(" = ");
                out.push_str(&format_expr(init));
            }
        }
        Stmt::Local {
            name,
            ty,
            dims: _,
            refinement,
            init,
            body,
            delocal_name,
            delocal_ty,
            delocal_dims: _,
            delocal_refinement,
            delocal,
        } => {
            push_indent(out, indent);
            out.push_str("local ");
            out.push_str(name);
            format_annotation_into(ty.as_ref(), refinement.as_ref(), out);
            out.push_str(" = ");
            out.push_str(&format_expr(init));
            out.push('\n');
            format_stmt_into(body, indent + 1, out);
            out.push('\n');
            push_indent(out, indent);
            out.push_str("delocal ");
            if let Some(delocal_ty) = delocal_ty {
                out.push_str(&format_spanned_type(delocal_ty));
                out.push(' ');
                out.push_str(delocal_name);
                if let Some(delocal_refinement) = delocal_refinement {
                    out.push_str(" where ");
                    out.push_str(&format_expr(delocal_refinement));
                }
            } else {
                out.push_str(delocal_name);
            }
            out.push_str(" = ");
            out.push_str(&format_expr(delocal));
        }
    }
}

fn format_annotation_into(
    ty: Option<&SpannedType>,
    refinement: Option<&SpannedExpr>,
    out: &mut String,
) {
    match (ty, refinement) {
        (None, None) => {}
        (Some(ty), None) => {
            out.push_str(": ");
            out.push_str(&format_spanned_type(ty));
        }
        (Some(ty), Some(refinement)) => {
            out.push_str(": ");
            out.push_str(&format_spanned_type(ty));
            out.push_str(" where ");
            out.push_str(&format_expr(refinement));
        }
        (None, Some(refinement)) => {
            out.push_str(": int where ");
            out.push_str(&format_expr(refinement));
        }
    }
}

fn format_spanned_type(ty: &SpannedType) -> String {
    format_type_expr(&ty.node)
}

fn is_default_iterate_step(expr: &SpannedExpr) -> bool {
    matches!(
        expr.node,
        Expr::Int {
            value: 1,
            unit: None
        }
    )
}

fn escape_string(value: &str) -> String {
    let mut escaped = String::new();
    for ch in value.chars() {
        match ch {
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\0' => escaped.push_str("\\0"),
            other => escaped.push(other),
        }
    }
    escaped
}

fn format_place(place: &Place) -> String {
    if place.indices.is_empty() {
        place.name.clone()
    } else {
        format!(
            "{}{}",
            place.name,
            place
                .indices
                .iter()
                .map(|index| format!("[{}]", format_expr(index)))
                .collect::<String>()
        )
    }
}

fn format_expr_with_prec(expr: &SpannedExpr, parent_prec: u8) -> String {
    let mut out = match &expr.node {
        Expr::Int { value, unit } => format_int(*value, unit.as_ref()),
        Expr::Bool(true) => "true".to_owned(),
        Expr::Bool(false) => "false".to_owned(),
        Expr::Array(elements) => format!(
            "[{}]",
            elements
                .iter()
                .map(format_expr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expr::Nil => "nil".to_owned(),
        Expr::Var(name) => name.clone(),
        Expr::Index { target, indices } => format!(
            "{target}{}",
            indices
                .iter()
                .map(|index| format!("[{}]", format_expr(index)))
                .collect::<String>()
        ),
        Expr::Empty { target } => format!("empty({target})"),
        Expr::Top { target } => format!("top({target})"),
        Expr::Size { target } => format!("size({target})"),
        Expr::Unary { op, expr } => {
            format!("{}{}", unary_op_text(*op), format_expr_with_prec(expr, 11))
        }
        Expr::Binary { op, left, right } => {
            let prec = binary_precedence(*op);
            let left = format_binary_child(left, prec, false);
            let right = format_binary_child(right, prec, true);
            format!("{left} {} {right}", binary_op_text(*op))
        }
    };

    if expr_precedence(expr) < parent_prec {
        out = format!("({out})");
    }

    out
}

fn format_binary_child(expr: &SpannedExpr, parent_prec: u8, right_child: bool) -> String {
    let child_prec = expr_precedence(expr);
    let mut out = format_expr_with_prec(expr, parent_prec);
    if right_child && child_prec == parent_prec {
        out = format!("({out})");
    }
    out
}

fn format_int(value: i64, unit: Option<&SpannedUnit>) -> String {
    let unit = unit.map(|unit| format!("<{}>", format_unit_expr(&unit.node)));
    let unit = unit.as_deref().unwrap_or("");

    match value {
        i64::MIN => format!("(0{unit} - 9223372036854775807{unit} - 1{unit})"),
        value if value < 0 => format!("(0{unit} - {}{unit})", -(value as i128)),
        value => format!("{value}{unit}"),
    }
}

fn expr_precedence(expr: &SpannedExpr) -> u8 {
    match &expr.node {
        Expr::Unary { .. } => 11,
        Expr::Binary { op, .. } => binary_precedence(*op),
        _ => 12,
    }
}

fn binary_precedence(op: BinaryOp) -> u8 {
    match op {
        BinaryOp::Or => 1,
        BinaryOp::And => 2,
        BinaryOp::BitOr => 3,
        BinaryOp::BitXor => 4,
        BinaryOp::BitAnd => 5,
        BinaryOp::Eq | BinaryOp::NotEq => 6,
        BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => 7,
        BinaryOp::Shl | BinaryOp::Shr => 8,
        BinaryOp::Add | BinaryOp::Sub => 9,
        BinaryOp::Mul | BinaryOp::FixedMul | BinaryOp::Div | BinaryOp::Rem => 10,
    }
}

fn format_unit_factor(factor: &UnitFactor, exponent: i64) -> String {
    match exponent {
        1 => factor.name.clone(),
        exponent => format!("{}^{exponent}", factor.name),
    }
}

fn update_op_text(op: UpdateOp) -> &'static str {
    match op {
        UpdateOp::Add => "+=",
        UpdateOp::Sub => "-=",
        UpdateOp::Xor => "^=",
    }
}

fn binary_op_text(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::FixedMul => "*/",
        BinaryOp::Div => "/",
        BinaryOp::Rem => "%",
        BinaryOp::Shl => "<<",
        BinaryOp::Shr => ">>",
        BinaryOp::BitAnd => "&",
        BinaryOp::BitXor => "^",
        BinaryOp::BitOr => "|",
        BinaryOp::Eq => "==",
        BinaryOp::NotEq => "!=",
        BinaryOp::Lt => "<",
        BinaryOp::LtEq => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::GtEq => ">=",
        BinaryOp::And => "&&",
        BinaryOp::Or => "||",
    }
}

fn unary_op_text(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "-",
        UnaryOp::Not => "!",
    }
}

fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str(INDENT);
    }
}

#[cfg(test)]
mod tests {
    use crate::{format_program, parse_program};

    #[test]
    fn formats_simple_sequence() {
        let program = parse_program("x += 1; y <=> z").expect("program parses");

        assert_eq!(format_program(&program), "x += 1;\ny <=> z\n");
    }

    #[test]
    fn formatted_output_parses_again() {
        let source = r#"
proc step(distance: int<m>, time: int<s>, speed: int<m/s>) {
  speed += distance / time
}

local acceleration: int<m/s^2> where acceleration >= 0<m/s^2> = 10<m/s^2>
  local xs: array<int<m>> = [1<m>, 2<m>]
    if xs[0] != xs[1] then
      xs[0] <=> xs[1]
    else
      skip
    fi xs[0] != xs[1]
  delocal xs = [2<m>, 1<m>]
delocal acceleration = 10<m/s^2>
"#;

        let program = parse_program(source).expect("program parses");
        let formatted = format_program(&program);

        parse_program(&formatted).expect("formatted program parses");
        insta::assert_snapshot!(formatted, @r###"
proc step(distance: int<m>, time: int<s>, speed: int<m/s>) {
  speed += distance / time
}

local acceleration: int<m/s^2> where acceleration >= 0<m/s^2> = 10<m/s^2>
  local xs: array<int<m>> = [1<m>, 2<m>]
    if xs[0] != xs[1] then
      xs[0] <=> xs[1]
    else
      skip
    fi xs[0] != xs[1]
  delocal xs = [2<m>, 1<m>]
delocal acceleration = 10<m/s^2>
"###);
    }
}
