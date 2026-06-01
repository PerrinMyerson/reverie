pub mod ast;
pub mod diagnostic;
mod formatter;
mod lexer;
mod parser;

pub use ast::{
    BinaryOp, Expr, GlobalDecl, Param, Place, Proc, Program, SourceSpan, Spanned, SpannedExpr,
    SpannedStmt, SpannedType, SpannedUnit, Stmt, TypeExpr, UnaryOp, UnitExpr, UnitFactor, UpdateOp,
};
pub use diagnostic::SyntaxDiagnostic;
pub use formatter::{format_expr, format_program, format_stmt, format_type_expr, format_unit_expr};
pub use lexer::{Token, lex};
pub use parser::{ParseOptions, parse_program, parse_program_with_options, parse_type_expr};
