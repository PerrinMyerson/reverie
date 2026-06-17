use chumsky::Stream;
use chumsky::prelude::*;

use crate::{
    BinaryOp, Expr, GlobalDecl, Param, Place, Proc, Program, SourceSpan, Spanned, SpannedExpr,
    SpannedStmt, SpannedType, SpannedUnit, Stmt, SyntaxDiagnostic, Token, TypeExpr, UnaryOp,
    UnitExpr, UnitFactor, UpdateOp, lex,
};

type ParseError = Simple<Token, SourceSpan>;

#[derive(Debug, Clone)]
struct DeclBinding {
    name: String,
    ty: SpannedType,
    len: usize,
    dims: Vec<usize>,
}

#[derive(Debug, Clone)]
struct LocalBinding {
    name: String,
    ty: Option<SpannedType>,
    dims: Vec<Option<usize>>,
    refinement: Option<SpannedExpr>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ParseOptions {
    pub legacy_janus: bool,
}

pub fn parse_program(source: &str) -> Result<Program, Vec<SyntaxDiagnostic>> {
    parse_program_with_options(source, ParseOptions::default())
}

pub fn parse_program_with_options(
    source: &str,
    options: ParseOptions,
) -> Result<Program, Vec<SyntaxDiagnostic>> {
    let tokens = normalize_tokens(source, lex(source)?, options);
    let tokens = normalize_legacy_bare_globals(tokens);
    parse_tokens(source, tokens, options)
}

pub fn parse_type_expr(source: &str) -> Result<SpannedType, Vec<SyntaxDiagnostic>> {
    let tokens = lex(source)?;
    parse_type_tokens(source, tokens)
}

fn parse_tokens(
    source: &str,
    tokens: Vec<Spanned<Token>>,
    options: ParseOptions,
) -> Result<Program, Vec<SyntaxDiagnostic>> {
    let end = source.len()..source.len();
    let stream = Stream::from_iter(
        end,
        tokens.into_iter().map(|token| (token.node, token.span)),
    );

    parser(options)
        .parse(stream)
        .map_err(|errors| errors.into_iter().map(to_diagnostic).collect())
}

fn normalize_tokens(
    source: &str,
    tokens: Vec<Spanned<Token>>,
    options: ParseOptions,
) -> Vec<Spanned<Token>> {
    if !options.legacy_janus {
        return tokens;
    }

    let tokens = strip_legacy_semicolon_comments(source, tokens);
    normalize_case_insensitive_idents(tokens)
}

fn strip_legacy_semicolon_comments(
    source: &str,
    tokens: Vec<Spanned<Token>>,
) -> Vec<Spanned<Token>> {
    let mut normalized = Vec::with_capacity(tokens.len());
    let mut comment_end = None;

    for token in tokens {
        if comment_end.is_some_and(|end| token.span.start < end) {
            continue;
        }
        comment_end = None;

        if token.node == Token::Semicolon {
            comment_end = Some(line_end(source, token.span.start));
        } else {
            normalized.push(token);
        }
    }

    normalized
}

fn line_end(source: &str, start: usize) -> usize {
    source[start..]
        .find('\n')
        .map(|offset| start + offset)
        .unwrap_or(source.len())
}

fn normalize_case_insensitive_idents(tokens: Vec<Spanned<Token>>) -> Vec<Spanned<Token>> {
    tokens
        .into_iter()
        .map(|token| {
            let node = match token.node {
                Token::Ident(name) => Token::Ident(name.to_ascii_lowercase()),
                other => other,
            };
            Spanned::new(node, token.span)
        })
        .collect()
}

fn normalize_legacy_bare_globals(tokens: Vec<Spanned<Token>>) -> Vec<Spanned<Token>> {
    let Some(proc_index) = tokens
        .iter()
        .position(|token| matches!(token.node, Token::Proc))
    else {
        return tokens;
    };

    if proc_index == 0 {
        return tokens;
    }

    let mut index = 0;
    let mut declarations = Vec::new();
    while index < proc_index {
        let Token::Ident(_) = &tokens[index].node else {
            return tokens;
        };

        let start = index;
        index += 1;
        while index < proc_index && matches!(tokens[index].node, Token::LBracket) {
            if index + 2 >= proc_index
                || !matches!(tokens[index + 1].node, Token::Int(_))
                || !matches!(tokens[index + 2].node, Token::RBracket)
            {
                return tokens;
            }
            index += 3;
        }
        declarations.push(start..index);
    }

    let mut normalized = Vec::with_capacity(tokens.len() + declarations.len() * 2);
    for declaration in declarations {
        let start = tokens[declaration.start].span.start;
        let end = tokens[declaration.end - 1].span.end;
        normalized.push(Spanned::new(Token::Global, start..start));
        normalized.extend(tokens[declaration].iter().cloned());
        normalized.push(Spanned::new(Token::Semicolon, end..end));
    }
    normalized.extend(tokens[proc_index..].iter().cloned());
    normalized
}

fn parse_type_tokens(
    source: &str,
    tokens: Vec<Spanned<Token>>,
) -> Result<SpannedType, Vec<SyntaxDiagnostic>> {
    let eof_span = source.len()..source.len();
    let stream = Stream::from_iter(
        eof_span,
        tokens.into_iter().map(|token| (token.node, token.span)),
    );

    type_expr_parser()
        .then_ignore(end())
        .parse(stream)
        .map_err(|errors| errors.into_iter().map(to_diagnostic).collect())
}

fn parser(options: ParseOptions) -> impl Parser<Token, Program, Error = ParseError> {
    global_parser(options)
        .repeated()
        .then(proc_parser(options).repeated())
        .then(stmt_parser(options).or_not())
        .then_ignore(end())
        .map(|((globals, procedures), body)| {
            let body = body.unwrap_or_else(|| {
                let entry = default_entry_name(&procedures);
                Spanned::new(
                    Stmt::Call {
                        name: entry.to_owned(),
                        args: Vec::new(),
                    },
                    0..0,
                )
            });
            let mut program = Program::with_globals_and_procedures(globals, procedures, body);
            hoist_main_declarations(&mut program);
            program
        })
}

fn default_entry_name(procedures: &[Proc]) -> &str {
    if procedures
        .iter()
        .any(|procedure| procedure.name == "main" && procedure.params.is_empty())
    {
        "main"
    } else if procedures
        .iter()
        .any(|procedure| procedure.name == "main_fwd" && procedure.params.is_empty())
    {
        "main_fwd"
    } else if procedures
        .iter()
        .any(|procedure| procedure.name == "test1" && procedure.params.is_empty())
    {
        "test1"
    } else if procedures
        .iter()
        .any(|procedure| procedure.name == "test" && procedure.params.is_empty())
    {
        "test"
    } else {
        "main"
    }
}

#[allow(clippy::result_large_err)]
fn global_parser(
    options: ParseOptions,
) -> impl Parser<Token, GlobalDecl, Error = ParseError> + Clone {
    let ident = select! { Token::Ident(name) => name };
    let expr = expr_parser(options);
    let dims = select! { Token::Int(value) => value }
        .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
        .delimited_by(just(Token::LBracket), just(Token::RBracket))
        .repeated();
    let ty = just(Token::Colon).ignore_then(type_expr_parser()).or_not();
    let init = just(Token::Eq).ignore_then(expr.clone()).or_not();
    let named_global =
        ident
            .then(dims)
            .then(ty)
            .then(init.clone())
            .map(|(((name, dims), ty), init)| {
                let len = leading_len(&dims);
                GlobalDecl {
                    name,
                    len,
                    dims,
                    ty,
                    init,
                    span: 0..0,
                }
            });
    let type_first_global =
        type_first_declaration_binding_parser()
            .then(init)
            .map(|(binding, init)| GlobalDecl {
                name: binding.name,
                len: binding.len,
                dims: binding.dims,
                ty: Some(binding.ty),
                init,
                span: 0..0,
            });

    just(Token::Global)
        .ignore_then(choice((type_first_global, named_global)))
        .then_ignore(just(Token::Semicolon).or_not())
        .map_with_span(|mut global, span| {
            global.span = span;
            global
        })
}

fn hoist_main_declarations(program: &mut Program) {
    let Stmt::Call { name, args } = &program.body.node else {
        return;
    };
    if name != "main" || !args.is_empty() {
        return;
    }

    let Some(main) = program
        .procedures
        .iter_mut()
        .find(|procedure| procedure.name == "main" && procedure.params.is_empty())
    else {
        return;
    };

    let body_span = main.body.span.clone();
    let previous_body = std::mem::replace(&mut main.body, synthetic_skip(body_span.clone()));
    let (globals, body) = split_leading_declarations(previous_body);
    if globals.is_empty() {
        main.body = body;
    } else {
        program.globals.extend(globals);
        main.body = body;
    }
}

fn split_leading_declarations(body: SpannedStmt) -> (Vec<GlobalDecl>, SpannedStmt) {
    match body.node {
        Stmt::Seq(statements) => {
            let mut globals = Vec::new();
            let mut rest = Vec::new();
            let mut in_prefix = true;

            for statement in statements {
                if in_prefix {
                    if let Some(global) = global_from_declaration(&statement) {
                        globals.push(global);
                        continue;
                    }
                    in_prefix = false;
                }
                rest.push(statement);
            }

            let span = statements_span(&rest);
            let body = match rest.len() {
                0 => synthetic_skip(body.span),
                1 => rest.into_iter().next().expect("one statement"),
                _ => Spanned::new(Stmt::Seq(rest), span),
            };
            (globals, body)
        }
        _ => {
            if let Some(global) = global_from_declaration(&body) {
                let span = body.span.clone();
                (vec![global], synthetic_skip(span))
            } else {
                (Vec::new(), body)
            }
        }
    }
}

fn global_from_declaration(statement: &SpannedStmt) -> Option<GlobalDecl> {
    let Stmt::Declare {
        name,
        ty,
        len,
        dims,
        init,
    } = &statement.node
    else {
        return None;
    };

    Some(GlobalDecl {
        name: name.clone(),
        len: *len,
        dims: dims.clone(),
        ty: Some(ty.clone()),
        init: init.clone(),
        span: statement.span.clone(),
    })
}

#[allow(clippy::result_large_err)]
fn proc_parser(options: ParseOptions) -> impl Parser<Token, Proc, Error = ParseError> + Clone {
    let ident = select! { Token::Ident(name) => name };
    let params = param_parser(options)
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .delimited_by(just(Token::LParen), just(Token::RParen));
    let params = params.or_not().map(|params| params.unwrap_or_default());
    let braced_body = stmt_parser(options).delimited_by(just(Token::LBrace), just(Token::RBrace));
    let body = choice((braced_body, stmt_parser(options)));

    just(Token::Proc)
        .ignore_then(ident)
        .then(params)
        .then(body)
        .map_with_span(|((name, params), body), span| Proc {
            name,
            params,
            body,
            span,
        })
}

#[allow(clippy::result_large_err)]
fn param_parser(options: ParseOptions) -> impl Parser<Token, Param, Error = ParseError> + Clone {
    let ident = select! { Token::Ident(name) => name };
    let expr = expr_parser(options);

    let reverie_param = ident
        .then(type_annotation(expr.clone()).or_not())
        .map_with_span(|(name, annotation), span| {
            let (ty, refinement) = annotation.unwrap_or((None, None));
            Param {
                name,
                ty,
                refinement,
                span,
            }
        });

    let janus_param =
        type_first_binding_parser(expr.clone()).map_with_span(|(name, ty, refinement), span| {
            Param {
                name,
                ty,
                refinement,
                span,
            }
        });

    choice((janus_param, reverie_param))
}

#[allow(clippy::result_large_err)]
fn stmt_parser(
    options: ParseOptions,
) -> impl Parser<Token, SpannedStmt, Error = ParseError> + Clone {
    let expr = expr_parser(options);
    let condition_expr = janus_condition_expr_parser(options);

    recursive(|stmt| {
        let ident = select! { Token::Ident(name) => name };
        let place = place_parser(expr.clone());

        let skip = just(Token::Skip).map_with_span(|_, span| Spanned::new(Stmt::Skip, span));

        let assert_stmt = just(Token::Assert)
            .ignore_then(condition_expr.clone())
            .map_with_span(|condition, span| Spanned::new(Stmt::Assert { condition }, span));

        let assert_helper_args = expr
            .clone()
            .then_ignore(just(Token::Comma))
            .then(expr.clone())
            .then_ignore(just(Token::Comma).or_not())
            .delimited_by(just(Token::LParen), just(Token::RParen));
        let assert_helper = choice((
            select! { Token::Ident(name) if name == "assert_eq" => BinaryOp::Eq },
            select! { Token::Ident(name) if name == "assert_ne" => BinaryOp::NotEq },
        ))
        .then(assert_helper_args)
        .map_with_span(|(op, (left, right)), span| {
            let condition = Spanned::new(
                Expr::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span.clone(),
            );
            Spanned::new(Stmt::Assert { condition }, span)
        });

        let update_op = choice((
            just(Token::PlusEq).to(UpdateOp::Add),
            just(Token::MinusEq).to(UpdateOp::Sub),
            just(Token::XorEq).to(UpdateOp::Xor),
            just(Token::BangEq).to(UpdateOp::Xor),
        ));

        let update = place
            .clone()
            .then(update_op)
            .then(expr.clone())
            .map_with_span(|((target, op), expr), span| {
                Spanned::new(Stmt::Update { target, op, expr }, span)
            });

        let increment_op = choice((
            just(Token::PlusPlus).map_with_span(|_, span| (UpdateOp::Add, span)),
            just(Token::MinusMinus).map_with_span(|_, span| (UpdateOp::Sub, span)),
        ));
        let postfix_increment = place.clone().then(increment_op.clone()).map_with_span(
            |(target, (op, expr_span)), span| {
                Spanned::new(
                    Stmt::Update {
                        target,
                        op,
                        expr: Spanned::new(
                            Expr::Int {
                                value: 1,
                                unit: None,
                            },
                            expr_span,
                        ),
                    },
                    span,
                )
            },
        );
        let prefix_increment =
            increment_op
                .then(place.clone())
                .map_with_span(|((op, expr_span), target), span| {
                    Spanned::new(
                        Stmt::Update {
                            target,
                            op,
                            expr: Spanned::new(
                                Expr::Int {
                                    value: 1,
                                    unit: None,
                                },
                                expr_span,
                            ),
                        },
                        span,
                    )
                });

        let swap = place
            .clone()
            .then_ignore(choice((just(Token::Swap), just(Token::Colon))))
            .then(place.clone())
            .map_with_span(|(left, right), span| Spanned::new(Stmt::Swap { left, right }, span));

        let swap_call_args = place
            .clone()
            .then_ignore(just(Token::Comma))
            .then(place.clone())
            .then_ignore(just(Token::Comma).or_not())
            .delimited_by(just(Token::LParen), just(Token::RParen));
        let swap_call = select! { Token::Ident(name) if name == "swap" => () }
            .ignore_then(swap_call_args)
            .map_with_span(|(left, right), span| Spanned::new(Stmt::Swap { left, right }, span));

        let stack_args = place
            .clone()
            .then_ignore(just(Token::Comma))
            .then(ident)
            .delimited_by(just(Token::LParen), just(Token::RParen));
        let push = just(Token::Push)
            .ignore_then(stack_args.clone())
            .map_with_span(|(source, stack), span| {
                Spanned::new(Stmt::Push { source, stack }, span)
            });
        let pop = just(Token::Pop)
            .ignore_then(stack_args)
            .map_with_span(|(target, stack), span| Spanned::new(Stmt::Pop { target, stack }, span));

        let args = place
            .clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .delimited_by(just(Token::LParen), just(Token::RParen));
        let args = args.or_not().map(|args| args.unwrap_or_default());

        let call = just(Token::Call)
            .ignore_then(ident)
            .then(args.clone())
            .map_with_span(|(name, args), span| Spanned::new(Stmt::Call { name, args }, span));

        let uncall = just(Token::Uncall)
            .ignore_then(ident)
            .then(args)
            .map_with_span(|(name, args), span| Spanned::new(Stmt::Uncall { name, args }, span));

        let read = just(Token::Read)
            .ignore_then(place.clone())
            .map_with_span(|target, span| Spanned::new(Stmt::Read { target }, span));

        let unread = just(Token::Unread)
            .ignore_then(place.clone())
            .map_with_span(|target, span| Spanned::new(Stmt::Unread { target }, span));

        let write = just(Token::Write)
            .ignore_then(place.clone())
            .map_with_span(|source, span| Spanned::new(Stmt::Write { source }, span));

        let unwrite = just(Token::Unwrite)
            .ignore_then(place.clone())
            .map_with_span(|source, span| Spanned::new(Stmt::Unwrite { source }, span));
        let show = just(Token::Show)
            .ignore_then(
                ident
                    .separated_by(just(Token::Comma))
                    .allow_trailing()
                    .at_least(1)
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map_with_span(|targets, span| Spanned::new(Stmt::Show { targets }, span));
        let printf_args = select! { Token::String(format) => format }
            .then(
                just(Token::Comma)
                    .ignore_then(
                        expr.clone()
                            .separated_by(just(Token::Comma))
                            .allow_trailing(),
                    )
                    .or_not()
                    .map(|args| args.unwrap_or_default()),
            )
            .delimited_by(just(Token::LParen), just(Token::RParen));
        let printf =
            just(Token::Printf)
                .ignore_then(printf_args)
                .map_with_span(|(format, args), span| {
                    Spanned::new(Stmt::Printf { format, args }, span)
                });

        let then_clause = just(Token::Then).ignore_then(stmt.clone()).or_not();
        let else_clause = just(Token::Else).ignore_then(stmt.clone()).or_not();
        let if_stmt = just(Token::If)
            .ignore_then(condition_expr.clone())
            .then(then_clause)
            .then(else_clause)
            .then_ignore(just(Token::Fi))
            .then(condition_expr.clone())
            .map_with_span(|(((entry, then_branch), else_branch), exit), span| {
                Spanned::new(
                    Stmt::If {
                        entry,
                        then_branch: Box::new(
                            then_branch.unwrap_or_else(|| synthetic_skip(span.clone())),
                        ),
                        else_branch: Box::new(
                            else_branch.unwrap_or_else(|| synthetic_skip(span.clone())),
                        ),
                        exit,
                    },
                    span,
                )
            });

        let do_clause = just(Token::Do).ignore_then(stmt.clone()).or_not();
        let loop_clause = just(Token::Loop).ignore_then(stmt.clone()).or_not();
        let loop_stmt = just(Token::From)
            .ignore_then(condition_expr.clone())
            .then(do_clause)
            .then(loop_clause)
            .then_ignore(just(Token::Until))
            .then(condition_expr.clone())
            .map_with_span(|(((entry, body), step), exit), span| {
                Spanned::new(
                    Stmt::Loop {
                        entry,
                        body: Box::new(body.unwrap_or_else(|| synthetic_skip(span.clone()))),
                        step: Box::new(step.unwrap_or_else(|| synthetic_skip(span.clone()))),
                        exit,
                    },
                    span,
                )
            });

        let iterate_step = just(Token::By).ignore_then(expr.clone()).or_not();
        let iterate_stmt = just(Token::Iterate)
            .ignore_then(just(Token::IntType))
            .ignore_then(ident)
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .then(iterate_step)
            .then_ignore(just(Token::To))
            .then(expr.clone())
            .then(stmt.clone())
            .then_ignore(just(Token::End))
            .map_with_span(|((((name, start), step), end), body), span| {
                Spanned::new(
                    Stmt::Iterate {
                        name,
                        start,
                        step: step.unwrap_or_else(|| default_iterate_step(span.clone())),
                        end,
                        body: Box::new(body),
                    },
                    span,
                )
            });

        let reverie_local_binding =
            ident
                .then(type_annotation(expr.clone()).or_not())
                .map(|(name, annotation)| {
                    let (ty, refinement) = annotation.unwrap_or((None, None));
                    LocalBinding {
                        name,
                        ty,
                        dims: Vec::new(),
                        refinement,
                    }
                });
        let local_binding = choice((
            type_first_local_binding_parser(expr.clone()),
            reverie_local_binding,
        ));
        let reverie_delocal_binding =
            ident
                .then(type_annotation(expr.clone()).or_not())
                .map(|(name, annotation)| {
                    let (ty, refinement) = annotation.unwrap_or((None, None));
                    LocalBinding {
                        name,
                        ty,
                        dims: Vec::new(),
                        refinement,
                    }
                });
        let delocal_binding = choice((
            type_first_local_binding_parser(expr.clone()),
            reverie_delocal_binding,
        ));

        let local_stmt = just(Token::Local)
            .ignore_then(local_binding)
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .then(stmt.clone())
            .then_ignore(just(Token::Delocal))
            .then(delocal_binding)
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .map_with_span(
                |((((local_binding, init), body), delocal_binding), delocal), span| {
                    Spanned::new(
                        Stmt::Local {
                            name: local_binding.name,
                            ty: local_binding.ty,
                            dims: local_binding.dims,
                            refinement: local_binding.refinement,
                            init,
                            body: Box::new(body),
                            delocal_name: delocal_binding.name,
                            delocal_ty: delocal_binding.ty,
                            delocal_dims: delocal_binding.dims,
                            delocal_refinement: delocal_binding.refinement,
                            delocal,
                        },
                        span,
                    )
                },
            );

        let declaration = type_first_declaration_binding_parser()
            .then(just(Token::Eq).ignore_then(expr.clone()).or_not())
            .map_with_span(|(binding, init), span| {
                Spanned::new(
                    Stmt::Declare {
                        name: binding.name,
                        ty: binding.ty,
                        len: binding.len,
                        dims: binding.dims,
                        init,
                    },
                    span,
                )
            });

        let item = choice((
            if_stmt,
            loop_stmt,
            iterate_stmt,
            local_stmt,
            declaration,
            push,
            pop,
            call,
            uncall,
            read,
            unread,
            write,
            unwrite,
            show,
            printf,
            prefix_increment,
            postfix_increment,
            update,
            swap_call,
            swap,
            assert_helper,
            assert_stmt,
            skip,
        ));

        item.then_ignore(just(Token::Semicolon).repeated())
            .repeated()
            .at_least(1)
            .map(|statements| {
                let span = statements_span(&statements);
                if statements.len() == 1 {
                    statements
                        .into_iter()
                        .next()
                        .expect("at least one statement")
                } else {
                    Spanned::new(Stmt::Seq(statements), span)
                }
            })
    })
}

#[allow(clippy::result_large_err)]
fn type_first_declaration_binding_parser()
-> impl Parser<Token, DeclBinding, Error = ParseError> + Clone {
    let ident = select! { Token::Ident(name) => name };
    let array_suffix = select! { Token::Int(value) => value }
        .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
        .delimited_by(just(Token::LBracket), just(Token::RBracket))
        .map_with_span(|len, span| (len, span))
        .repeated();

    type_expr_parser()
        .then(ident)
        .then(array_suffix)
        .map(|((ty, name), suffixes)| {
            let spans = suffixes
                .iter()
                .map(|(_, span)| span.clone())
                .collect::<Vec<_>>();
            let dims = suffixes.iter().map(|(len, _)| *len).collect::<Vec<_>>();
            let len = leading_len(&dims);
            DeclBinding {
                name,
                ty: apply_array_suffixes(ty, &spans),
                len,
                dims,
            }
        })
}

#[allow(clippy::result_large_err)]
fn type_first_binding_parser(
    expr: impl Parser<Token, SpannedExpr, Error = ParseError> + Clone,
) -> impl Parser<Token, (String, Option<SpannedType>, Option<SpannedExpr>), Error = ParseError> + Clone
{
    let ident = select! { Token::Ident(name) => name };
    let array_suffix = just(Token::LBracket)
        .then_ignore(just(Token::RBracket))
        .map_with_span(|_, span| span)
        .repeated();

    type_expr_parser()
        .then(ident)
        .then(array_suffix)
        .then(just(Token::Where).ignore_then(expr).or_not())
        .map(|(((ty, name), suffixes), refinement)| {
            let ty = apply_array_suffixes(ty, &suffixes);
            (name, Some(ty), refinement)
        })
}

#[allow(clippy::result_large_err)]
fn type_first_local_binding_parser(
    expr: impl Parser<Token, SpannedExpr, Error = ParseError> + Clone,
) -> impl Parser<Token, LocalBinding, Error = ParseError> + Clone {
    let ident = select! { Token::Ident(name) => name };
    let array_suffix = select! { Token::Int(value) => value }
        .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
        .or_not()
        .delimited_by(just(Token::LBracket), just(Token::RBracket))
        .map_with_span(|len, span| (len, span))
        .repeated();

    type_expr_parser()
        .then(ident)
        .then(array_suffix)
        .then(just(Token::Where).ignore_then(expr).or_not())
        .map(|(((ty, name), suffixes), refinement)| {
            let spans = suffixes
                .iter()
                .map(|(_, span)| span.clone())
                .collect::<Vec<_>>();
            let dims = suffixes.iter().map(|(len, _)| *len).collect::<Vec<_>>();
            let ty = apply_array_suffixes(ty, &spans);
            LocalBinding {
                name,
                ty: Some(ty),
                dims,
                refinement,
            }
        })
}

fn type_annotation(
    expr: impl Parser<Token, SpannedExpr, Error = ParseError> + Clone,
) -> impl Parser<Token, (Option<SpannedType>, Option<SpannedExpr>), Error = ParseError> + Clone {
    just(Token::Colon)
        .ignore_then(type_expr_parser())
        .map(Some)
        .then(just(Token::Where).ignore_then(expr).or_not())
}

#[allow(clippy::result_large_err)]
fn type_expr_parser() -> impl Parser<Token, SpannedType, Error = ParseError> + Clone {
    recursive(|ty| {
        let int = just(Token::IntType)
            .then(unit_suffix().or_not())
            .map_with_span(|(_, unit), span| Spanned::new(TypeExpr::Int { unit }, span));

        let bool =
            just(Token::BoolType).map_with_span(|_, span| Spanned::new(TypeExpr::Bool, span));

        let array = just(Token::ArrayType)
            .ignore_then(ty.clone().delimited_by(just(Token::Lt), just(Token::Gt)))
            .map_with_span(|element, span| {
                Spanned::new(
                    TypeExpr::Array {
                        element: Box::new(element),
                    },
                    span,
                )
            });

        let tensor_shape = just(Token::Comma)
            .ignore_then(select! { Token::Int(value) => value })
            .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
            .repeated()
            .at_least(1);
        let tensor = select! { Token::Ident(name) if name.eq_ignore_ascii_case("tensor") => () }
            .ignore_then(
                ty.clone()
                    .then(tensor_shape)
                    .delimited_by(just(Token::Lt), just(Token::Gt)),
            )
            .map_with_span(|(element, shape), span| {
                Spanned::new(
                    TypeExpr::Tensor {
                        element: Box::new(element),
                        shape,
                    },
                    span,
                )
            });

        let witness = select! { Token::Ident(name) if name.eq_ignore_ascii_case("witness") => () }
            .ignore_then(ty.clone().delimited_by(just(Token::Lt), just(Token::Gt)))
            .map_with_span(|inner, span| {
                Spanned::new(
                    TypeExpr::Witness {
                        inner: Box::new(inner),
                    },
                    span,
                )
            });

        let stack =
            just(Token::StackType).map_with_span(|_, span| Spanned::new(TypeExpr::Stack, span));

        choice((witness, tensor, array, int, bool, stack))
    })
}

fn apply_array_suffixes(mut ty: SpannedType, suffixes: &[SourceSpan]) -> SpannedType {
    for suffix in suffixes {
        let span = ty.span.start..suffix.end;
        ty = Spanned::new(
            TypeExpr::Array {
                element: Box::new(ty),
            },
            span,
        );
    }

    ty
}

fn leading_len(dims: &[usize]) -> usize {
    dims.first().copied().unwrap_or(1)
}

fn unit_suffix() -> impl Parser<Token, SpannedUnit, Error = ParseError> + Clone {
    unit_parser().delimited_by(just(Token::Lt), just(Token::Gt))
}

#[allow(clippy::result_large_err)]
fn unit_parser() -> impl Parser<Token, SpannedUnit, Error = ParseError> + Clone {
    let ident = select! { Token::Ident(name) => name };
    let exponent =
        select! { Token::Int(value) if value <= i32::MAX as i64 => value as i32 }.boxed();
    let factor = ident
        .then(just(Token::Caret).ignore_then(exponent).or_not())
        .map(|(name, exponent)| UnitFactor {
            name,
            exponent: exponent.unwrap_or(1),
        })
        .boxed();

    factor
        .clone()
        .then(
            choice((just(Token::Star).to(1), just(Token::Slash).to(-1)))
                .then(factor)
                .repeated(),
        )
        .map_with_span(|(first, rest), span| {
            let mut factors = vec![first];
            factors.extend(rest.into_iter().map(|(sign, mut factor)| {
                factor.exponent *= sign;
                factor
            }));

            Spanned::new(UnitExpr::new(factors), span)
        })
        .boxed()
}

#[allow(clippy::result_large_err)]
fn expr_parser(
    options: ParseOptions,
) -> impl Parser<Token, SpannedExpr, Error = ParseError> + Clone {
    expr_parser_with_options(false, options)
}

#[allow(clippy::result_large_err)]
fn janus_condition_expr_parser(
    options: ParseOptions,
) -> impl Parser<Token, SpannedExpr, Error = ParseError> + Clone {
    expr_parser_with_options(true, options)
}

#[allow(clippy::result_large_err)]
fn expr_parser_with_options(
    allow_janus_eq_alias: bool,
    _options: ParseOptions,
) -> impl Parser<Token, SpannedExpr, Error = ParseError> + Clone {
    modern_expr_parser_with(allow_janus_eq_alias)
}

#[allow(clippy::result_large_err)]
fn modern_expr_parser_with(
    allow_janus_eq_alias: bool,
) -> impl Parser<Token, SpannedExpr, Error = ParseError> + Clone {
    recursive(|expr| {
        let ident = select! { Token::Ident(name) => name };
        let indexed_var = select! { Token::Ident(name) => name }
            .then(
                expr.clone()
                    .delimited_by(just(Token::LBracket), just(Token::RBracket))
                    .repeated(),
            )
            .map_with_span(|(name, indices), span| {
                Spanned::new(
                    if indices.is_empty() {
                        Expr::Var(name)
                    } else {
                        Expr::Index {
                            target: name,
                            indices,
                        }
                    },
                    span,
                )
            });
        let size_expr = just(Token::Size)
            .ignore_then(ident.delimited_by(just(Token::LParen), just(Token::RParen)))
            .map_with_span(|target, span| Spanned::new(Expr::Size { target }, span));
        let len_expr = select! { Token::Ident(name) if name == "len" => () }
            .ignore_then(ident.delimited_by(just(Token::LParen), just(Token::RParen)))
            .map_with_span(|target, span| Spanned::new(Expr::Size { target }, span));
        let empty_expr = just(Token::Empty)
            .ignore_then(ident.delimited_by(just(Token::LParen), just(Token::RParen)))
            .map_with_span(|target, span| Spanned::new(Expr::Empty { target }, span));
        let is_empty_expr = select! { Token::Ident(name) if name == "is_empty" => () }
            .ignore_then(ident.delimited_by(just(Token::LParen), just(Token::RParen)))
            .map_with_span(|target, span| Spanned::new(Expr::Empty { target }, span));
        let top_expr = just(Token::Top)
            .ignore_then(ident.delimited_by(just(Token::LParen), just(Token::RParen)))
            .map_with_span(|target, span| Spanned::new(Expr::Top { target }, span));
        let peek_expr = select! { Token::Ident(name) if name == "peek" => () }
            .ignore_then(ident.delimited_by(just(Token::LParen), just(Token::RParen)))
            .map_with_span(|target, span| Spanned::new(Expr::Top { target }, span));
        let call_expr = select! { Token::Ident(name) => name }
            .then(
                expr.clone()
                    .separated_by(just(Token::Comma))
                    .allow_trailing()
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map_with_span(|(name, args), span| Spanned::new(Expr::Call { name, args }, span));

        let atom = choice((
            select! { Token::Int(value) => value }
                .then(unit_suffix().or_not())
                .map_with_span(|(value, unit), span| Spanned::new(Expr::Int { value, unit }, span)),
            just(Token::True).map_with_span(|_, span| Spanned::new(Expr::Bool(true), span)),
            just(Token::False).map_with_span(|_, span| Spanned::new(Expr::Bool(false), span)),
            just(Token::Nil).map_with_span(|_, span| Spanned::new(Expr::Nil, span)),
            expr.clone()
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .delimited_by(just(Token::LBracket), just(Token::RBracket))
                .map_with_span(|elements, span| Spanned::new(Expr::Array(elements), span)),
            expr.clone()
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .delimited_by(just(Token::LBrace), just(Token::RBrace))
                .map_with_span(|elements, span| Spanned::new(Expr::Array(elements), span)),
            empty_expr,
            is_empty_expr,
            top_expr,
            peek_expr,
            len_expr,
            size_expr,
            call_expr,
            indexed_var,
            just(Token::LParen)
                .ignore_then(expr.clone())
                .then_ignore(just(Token::RParen)),
        ));

        let unary_op = choice((
            just(Token::Minus).to(UnaryOp::Neg),
            just(Token::Bang).to(UnaryOp::Not),
        ))
        .boxed();

        let unary = unary_op
            .repeated()
            .then(atom.clone())
            .map_with_span(|(ops, expr), span| {
                ops.into_iter().rev().fold(expr, |expr, op| {
                    Spanned::new(
                        Expr::Unary {
                            op,
                            expr: Box::new(expr),
                        },
                        span.clone(),
                    )
                })
            })
            .boxed();

        let product = unary
            .clone()
            .then(
                choice((
                    just(Token::Star).to(BinaryOp::Mul),
                    just(Token::StarSlash).to(BinaryOp::FixedMul),
                    just(Token::Slash).to(BinaryOp::Div),
                    just(Token::Percent).to(BinaryOp::Rem),
                ))
                .then(unary)
                .repeated(),
            )
            .foldl(binary_expr)
            .boxed();

        let sum = product
            .clone()
            .then(
                choice((
                    just(Token::Plus).to(BinaryOp::Add),
                    just(Token::Minus).to(BinaryOp::Sub),
                ))
                .then(product)
                .repeated(),
            )
            .foldl(binary_expr)
            .boxed();

        let shift = sum
            .clone()
            .then(
                choice((
                    just(Token::Lt)
                        .then_ignore(just(Token::Lt))
                        .to(BinaryOp::Shl),
                    just(Token::Gt)
                        .then_ignore(just(Token::Gt))
                        .to(BinaryOp::Shr),
                ))
                .then(sum)
                .repeated(),
            )
            .foldl(binary_expr)
            .boxed();

        let comparison = shift
            .clone()
            .then(
                choice((
                    just(Token::LtEq).to(BinaryOp::LtEq),
                    just(Token::GtEq).to(BinaryOp::GtEq),
                    just(Token::Lt).to(BinaryOp::Lt),
                    just(Token::Gt).to(BinaryOp::Gt),
                ))
                .then(shift)
                .repeated(),
            )
            .foldl(binary_expr)
            .boxed();

        let equality_op = if allow_janus_eq_alias {
            choice((
                just(Token::EqEq).to(BinaryOp::Eq),
                just(Token::Eq).to(BinaryOp::Eq),
                just(Token::BangEq).to(BinaryOp::NotEq),
            ))
            .boxed()
        } else {
            choice((
                just(Token::EqEq).to(BinaryOp::Eq),
                just(Token::BangEq).to(BinaryOp::NotEq),
            ))
            .boxed()
        };

        let equality = comparison
            .clone()
            .then(equality_op.then(comparison).repeated())
            .foldl(binary_expr)
            .boxed();

        let bit_and = equality
            .clone()
            .then(
                just(Token::Amp)
                    .to(BinaryOp::BitAnd)
                    .then(equality)
                    .repeated(),
            )
            .foldl(binary_expr)
            .boxed();

        let bit_xor = bit_and
            .clone()
            .then(
                choice((
                    just(Token::Caret).to(BinaryOp::BitXor),
                    just(Token::Bang).to(BinaryOp::BitXor),
                ))
                .then(bit_and)
                .repeated(),
            )
            .foldl(binary_expr)
            .boxed();

        let bit_or = bit_xor
            .clone()
            .then(
                just(Token::Pipe)
                    .to(BinaryOp::BitOr)
                    .then(bit_xor)
                    .repeated(),
            )
            .foldl(binary_expr)
            .boxed();

        let and = bit_or
            .clone()
            .then(
                just(Token::AndAnd)
                    .to(BinaryOp::And)
                    .then(bit_or)
                    .repeated(),
            )
            .foldl(binary_expr)
            .boxed();

        and.clone()
            .then(just(Token::OrOr).to(BinaryOp::Or).then(and).repeated())
            .foldl(binary_expr)
            .boxed()
    })
}

#[allow(clippy::result_large_err)]
fn place_parser(
    expr: impl Parser<Token, SpannedExpr, Error = ParseError> + Clone,
) -> impl Parser<Token, Place, Error = ParseError> + Clone {
    select! { Token::Ident(name) => name }
        .then(
            expr.delimited_by(just(Token::LBracket), just(Token::RBracket))
                .repeated(),
        )
        .map(|(name, indices)| Place::with_indices(name, indices))
}

fn binary_expr(left: SpannedExpr, (op, right): (BinaryOp, SpannedExpr)) -> SpannedExpr {
    let span = left.span.start..right.span.end;
    Spanned::new(
        Expr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        },
        span,
    )
}

fn synthetic_skip(span: SourceSpan) -> SpannedStmt {
    Spanned::new(Stmt::Skip, span)
}

fn default_iterate_step(span: SourceSpan) -> SpannedExpr {
    Spanned::new(
        Expr::Int {
            value: 1,
            unit: None,
        },
        span.start..span.start,
    )
}

fn statements_span(statements: &[SpannedStmt]) -> SourceSpan {
    let start = statements
        .first()
        .map(|statement| statement.span.start)
        .unwrap_or(0);
    let end = statements
        .last()
        .map(|statement| statement.span.end)
        .unwrap_or(start);

    start..end
}

fn to_diagnostic(error: ParseError) -> SyntaxDiagnostic {
    let expected = expected_tokens(&error);
    let found = error
        .found()
        .map(|token| token.display())
        .unwrap_or_else(|| "end of input".to_owned());

    let message = if expected.is_empty() {
        format!("unexpected {found}")
    } else {
        format!("expected {expected}, found {found}")
    };

    SyntaxDiagnostic::new(message, error.span(), "the parser got stuck here")
}

fn expected_tokens(error: &ParseError) -> String {
    let mut expected = error
        .expected()
        .map(|expected| {
            expected
                .as_ref()
                .map(|token| token.display())
                .unwrap_or_else(|| "end of input".to_owned())
        })
        .collect::<Vec<_>>();

    expected.sort();
    expected.dedup();

    match expected.as_slice() {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} or {last}", rest.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::{BinaryOp, Place, Stmt, TypeExpr, UpdateOp};

    #[test]
    fn parses_skip() {
        let program = parse_program("skip").expect("program parses");
        assert_eq!(program.body.node, Stmt::Skip);
        assert_eq!(program.body.span, 0..4);
    }

    #[test]
    fn parses_skip_with_trailing_semicolon() {
        let program = parse_program("skip;").expect("program parses");
        assert_eq!(program.body.node, Stmt::Skip);
        assert_eq!(program.body.span, 0..4);
    }

    #[test]
    fn parses_sequence() {
        let program = parse_program("skip; skip").expect("program parses");
        assert!(matches!(program.body.node, Stmt::Seq(_)));
        assert_eq!(program.body.span, 0..10);
    }

    #[test]
    fn parses_semicolonless_statement_sequence() {
        let program = parse_program(
            r#"
x += 1
y += 2
x <=> y
"#,
        )
        .expect("program parses");

        let Stmt::Seq(statements) = program.body.node else {
            panic!("expected sequence");
        };
        assert_eq!(statements.len(), 3);
    }

    #[test]
    fn ignores_block_comments() {
        let program = parse_program("/* Janus block comment */ skip").expect("program parses");
        assert_eq!(program.body.node, Stmt::Skip);
    }

    #[test]
    fn ignores_attached_janus_semicolon_comments() {
        let program = parse_program(
            r#"
num ;Number to transform.
procedure main
  num += 1;increment once
  num -= 1
"#,
        )
        .expect("program parses");
        assert_eq!(program.globals[0].name, "num");
        assert_eq!(program.procedures[0].name, "main");
    }

    #[test]
    fn accepts_case_insensitive_legacy_keywords() {
        let program = parse_program(
            r#"
num ;Number to transform.
PrOcEdUrE main
  IF num = 0 THEN
    num += 1
  FI num = 1
"#,
        )
        .expect("program parses");
        assert_eq!(program.globals[0].name, "num");
        assert_eq!(program.procedures[0].name, "main");
    }

    #[test]
    fn legacy_janus_mode_treats_semicolons_as_comments() {
        let program = parse_program_with_options(
            r#"
procedure main
  x += 1; y += 2
"#,
            ParseOptions { legacy_janus: true },
        )
        .expect("program parses");

        let Stmt::Update { target, expr, .. } = &program.procedures[0].body.node else {
            panic!("expected a single update");
        };
        assert_eq!(target.name, "x");
        assert!(matches!(expr.node, Expr::Int { value: 1, .. }));
    }

    #[test]
    fn legacy_janus_mode_normalizes_identifiers() {
        let program = parse_program_with_options(
            r#"
COUNT
PrOcEdUrE Main
  IF COUNT = 0 THEN
    Count += 1
  FI count = 1
"#,
            ParseOptions { legacy_janus: true },
        )
        .expect("program parses");

        assert_eq!(program.globals[0].name, "count");
        assert_eq!(program.procedures[0].name, "main");
        let Stmt::If {
            then_branch, exit, ..
        } = &program.procedures[0].body.node
        else {
            panic!("expected if statement");
        };
        let Stmt::Update { target, .. } = &then_branch.node else {
            panic!("expected update");
        };
        assert_eq!(target.name, "count");
        assert!(matches!(
            exit.node,
            Expr::Binary {
                op: BinaryOp::Eq,
                ..
            }
        ));
    }

    #[test]
    fn legacy_janus_mode_keeps_modern_expression_precedence() {
        let program = parse_program_with_options(
            "procedure main\n  assert 1 + 2 * 3 = 9",
            ParseOptions { legacy_janus: true },
        )
        .expect("program parses");

        let Stmt::Assert { condition } = &program.procedures[0].body.node else {
            panic!("expected assert");
        };
        let Expr::Binary {
            op: BinaryOp::Eq,
            left,
            ..
        } = &condition.node
        else {
            panic!("expected equality");
        };
        assert!(matches!(
            left.node,
            Expr::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn parses_assert() {
        let program = parse_program("assert x == 1").expect("program parses");
        let Stmt::Assert { condition } = program.body.node else {
            panic!("expected assert");
        };

        assert!(matches!(
            condition.node,
            Expr::Binary {
                op: BinaryOp::Eq,
                ..
            }
        ));
    }

    #[test]
    fn parses_assert_helpers_as_assertions() {
        let program = parse_program("assert_eq(x, 1); assert_ne(y, 2)").expect("program parses");
        let Stmt::Seq(statements) = program.body.node else {
            panic!("expected sequence");
        };

        let Stmt::Assert { condition } = &statements[0].node else {
            panic!("expected equality assert");
        };
        assert!(matches!(
            condition.node,
            Expr::Binary {
                op: BinaryOp::Eq,
                ..
            }
        ));

        let Stmt::Assert { condition } = &statements[1].node else {
            panic!("expected inequality assert");
        };
        assert!(matches!(
            condition.node,
            Expr::Binary {
                op: BinaryOp::NotEq,
                ..
            }
        ));
    }

    #[test]
    fn parses_unary_not() {
        let program = parse_program("assert !(x == 1)").expect("program parses");
        let Stmt::Assert { condition } = program.body.node else {
            panic!("expected assert");
        };

        assert!(matches!(
            condition.node,
            Expr::Unary {
                op: UnaryOp::Not,
                ..
            }
        ));
    }

    #[test]
    fn parses_unary_numeric_negation() {
        let program = parse_program("assert -x == -1").expect("program parses");
        let Stmt::Assert { condition } = program.body.node else {
            panic!("expected assert");
        };
        let Expr::Binary { left, right, .. } = condition.node else {
            panic!("expected equality");
        };

        assert!(matches!(
            left.node,
            Expr::Unary {
                op: UnaryOp::Neg,
                ..
            }
        ));
        assert!(matches!(
            right.node,
            Expr::Unary {
                op: UnaryOp::Neg,
                ..
            }
        ));
    }

    #[test]
    fn parses_janus_operator_aliases() {
        let program = parse_program(
            "assert (x = 5) & ~(y # 3) | false; z += x \\ y; z += x ! y; x != y; x : y",
        )
        .expect("program parses");

        let Stmt::Seq(statements) = program.body.node else {
            panic!("expected sequence");
        };
        assert_eq!(statements.len(), 5);
        assert!(matches!(
            statements[3].node,
            Stmt::Update {
                op: UpdateOp::Xor,
                ..
            }
        ));
        assert!(matches!(statements[4].node, Stmt::Swap { .. }));
    }

    #[test]
    fn parses_bitwise_and_shift_operators() {
        let program =
            parse_program("x += (a & 1) | (b ^ 2); y += 1 << 3; z += y >> 1; w += y */ z")
                .expect("program parses");

        let Stmt::Seq(statements) = program.body.node else {
            panic!("expected sequence");
        };
        assert_eq!(statements.len(), 4);
        let Stmt::Update { expr, .. } = &statements[0].node else {
            panic!("expected update");
        };
        assert!(matches!(
            expr.node,
            Expr::Binary {
                op: BinaryOp::BitOr,
                ..
            }
        ));
        let Stmt::Update { expr, .. } = &statements[1].node else {
            panic!("expected update");
        };
        assert!(matches!(
            expr.node,
            Expr::Binary {
                op: BinaryOp::Shl,
                ..
            }
        ));
        let Stmt::Update { expr, .. } = &statements[2].node else {
            panic!("expected update");
        };
        assert!(matches!(
            expr.node,
            Expr::Binary {
                op: BinaryOp::Shr,
                ..
            }
        ));
        let Stmt::Update { expr, .. } = &statements[3].node else {
            panic!("expected update");
        };
        assert!(matches!(
            expr.node,
            Expr::Binary {
                op: BinaryOp::FixedMul,
                ..
            }
        ));
    }

    #[test]
    fn parses_janus_optional_control_clauses() {
        let program = parse_program(
            "if x = 0 then x += 1 fi x # 0; if y # 0 else y += 1 fi y = 0; from i = 0 loop i += 1 until i = 1",
        )
        .expect("program parses");

        let Stmt::Seq(statements) = program.body.node else {
            panic!("expected sequence");
        };
        assert_eq!(statements.len(), 3);

        let Stmt::If { else_branch, .. } = &statements[0].node else {
            panic!("expected if");
        };
        assert!(matches!(else_branch.node, Stmt::Skip));

        let Stmt::If { then_branch, .. } = &statements[1].node else {
            panic!("expected if");
        };
        assert!(matches!(then_branch.node, Stmt::Skip));

        let Stmt::Loop { body, .. } = &statements[2].node else {
            panic!("expected loop");
        };
        assert!(matches!(body.node, Stmt::Skip));
    }

    #[test]
    fn parses_janus_iterate_loop() {
        let program = parse_program(
            r#"
iterate int i = n - 1 by -1 to 0
  x += i
end
"#,
        )
        .expect("program parses");

        let Stmt::Iterate {
            name,
            start,
            step,
            end,
            body,
        } = program.body.node
        else {
            panic!("expected iterate");
        };
        assert_eq!(name, "i");
        assert!(matches!(start.node, Expr::Binary { .. }));
        assert!(matches!(
            step.node,
            Expr::Unary {
                op: UnaryOp::Neg,
                ..
            }
        ));
        assert!(matches!(end.node, Expr::Int { value: 0, .. }));
        assert!(matches!(body.node, Stmt::Update { .. }));
    }

    #[test]
    fn parses_size_expression() {
        let program = parse_program("assert size(xs) == 3").expect("program parses");
        let Stmt::Assert { condition } = program.body.node else {
            panic!("expected assert");
        };
        let Expr::Binary { left, .. } = condition.node else {
            panic!("expected equality");
        };

        assert!(matches!(
            left.node,
            Expr::Size { ref target } if target == "xs"
        ));
    }

    #[test]
    fn parses_len_expression_as_size_alias() {
        let program = parse_program("assert len(xs) == 3").expect("program parses");
        let Stmt::Assert { condition } = program.body.node else {
            panic!("expected assert");
        };
        let Expr::Binary { left, .. } = condition.node else {
            panic!("expected equality");
        };

        assert!(matches!(
            left.node,
            Expr::Size { ref target } if target == "xs"
        ));
    }

    #[test]
    fn parses_stack_operations_and_expressions() {
        let program = parse_program(
            "local s: stack = nil push(x, s); assert !empty(s); assert top(s) == 1; pop(x, s) delocal s = nil",
        )
        .expect("program parses");

        let Stmt::Local { ty, body, .. } = program.body.node else {
            panic!("expected local stack");
        };
        assert!(matches!(
            ty.expect("stack annotation").node,
            TypeExpr::Stack
        ));

        let Stmt::Seq(statements) = body.node else {
            panic!("expected sequence");
        };
        assert!(matches!(statements[0].node, Stmt::Push { .. }));
        assert!(matches!(statements[3].node, Stmt::Pop { .. }));
    }

    #[test]
    fn parses_stack_readability_aliases() {
        let program =
            parse_program("assert !is_empty(s); assert peek(s) == 1").expect("program parses");
        let Stmt::Seq(statements) = program.body.node else {
            panic!("expected sequence");
        };

        let Stmt::Assert { condition } = &statements[0].node else {
            panic!("expected first assert");
        };
        assert!(matches!(
            condition.node,
            Expr::Unary {
                op: UnaryOp::Not,
                ..
            }
        ));

        let Stmt::Assert { condition } = &statements[1].node else {
            panic!("expected second assert");
        };
        let Expr::Binary { left, .. } = &condition.node else {
            panic!("expected equality");
        };
        assert!(matches!(
            left.node,
            Expr::Top { ref target } if target == "s"
        ));
    }

    #[test]
    fn parses_bool_type_annotations() {
        let program = parse_program(
            "global flag: bool = true; local ok: bool = false skip delocal ok = false",
        )
        .expect("program parses");
        assert_eq!(program.globals.len(), 1);
        assert!(matches!(
            program.globals[0]
                .ty
                .as_ref()
                .expect("bool annotation")
                .node,
            TypeExpr::Bool
        ));
        let ty = parse_type_expr("array<bool>").expect("type parses");
        assert!(matches!(ty.node, TypeExpr::Array { .. }));
    }

    #[test]
    fn parses_tensor_type_annotations_and_builtin_calls() {
        let ty = parse_type_expr("tensor<int, 2, 3>").expect("type parses");
        let TypeExpr::Tensor { element, shape } = ty.node else {
            panic!("expected tensor type");
        };
        assert!(matches!(element.node, TypeExpr::Int { .. }));
        assert_eq!(shape, vec![2, 3]);

        let program = parse_program(
            "local out: tensor<int, 2, 2> = [[0, 0], [0, 0]]
               out += matmul(a, b)
             delocal out = [[58, 64], [139, 154]]",
        )
        .expect("program parses");
        let Stmt::Local { ty, body, .. } = program.body.node else {
            panic!("expected local tensor");
        };
        assert!(matches!(
            ty.expect("tensor annotation").node,
            TypeExpr::Tensor { .. }
        ));
        let Stmt::Update { expr, .. } = body.node else {
            panic!("expected tensor update");
        };
        assert!(matches!(
            expr.node,
            Expr::Call { ref name, .. } if name == "matmul"
        ));
    }

    #[test]
    fn parses_witness_type_annotations() {
        let ty = parse_type_expr("witness<tensor<int, 2, 3>>").expect("type parses");
        let TypeExpr::Witness { inner } = ty.node else {
            panic!("expected witness type");
        };
        let TypeExpr::Tensor { element, shape } = inner.node else {
            panic!("expected witness tensor payload");
        };
        assert!(matches!(element.node, TypeExpr::Int { .. }));
        assert_eq!(shape, vec![2, 3]);

        let program = parse_program(
            "global trace: witness<tensor<int, 2>>;
             trace += [1, 2]",
        )
        .expect("program parses");
        assert!(matches!(
            program.globals[0]
                .ty
                .as_ref()
                .expect("type annotation")
                .node,
            TypeExpr::Witness { .. }
        ));
    }

    #[test]
    fn parses_janus_show_statement() {
        let program = parse_program("show(s, t)").expect("program parses");

        let Stmt::Show { targets } = program.body.node else {
            panic!("expected show");
        };
        assert_eq!(targets, ["s", "t"]);
    }

    #[test]
    fn parses_janus_printf_statement() {
        let program = parse_program(r#"printf("%d %d\n", x, xs[0])"#).expect("program parses");

        let Stmt::Printf { format, args } = program.body.node else {
            panic!("expected printf");
        };
        assert_eq!(format, "%d %d\n");
        assert_eq!(args.len(), 2);
        assert!(matches!(args[1].node, Expr::Index { .. }));
    }

    #[test]
    fn parses_janus_type_first_parameters_and_array_suffixes() {
        let program =
            parse_program("procedure move_stack(stack src, stack dst, int n, int fact[]) { skip }")
                .expect("program parses");

        let procedure = &program.procedures[0];
        assert_eq!(procedure.params.len(), 4);
        assert_eq!(procedure.params[0].name, "src");
        assert!(matches!(
            procedure.params[0]
                .ty
                .as_ref()
                .expect("stack annotation")
                .node,
            TypeExpr::Stack
        ));
        assert_eq!(procedure.params[2].name, "n");
        assert!(matches!(
            procedure.params[2]
                .ty
                .as_ref()
                .expect("int annotation")
                .node,
            TypeExpr::Int { .. }
        ));
        assert_eq!(procedure.params[3].name, "fact");
        assert!(matches!(
            procedure.params[3]
                .ty
                .as_ref()
                .expect("array annotation")
                .node,
            TypeExpr::Array { .. }
        ));
    }

    #[test]
    fn parses_janus_type_first_local_and_delocal() {
        let program =
            parse_program("local int xs[3] = [1, 2, 3] skip delocal int xs[3] = [1, 2, 3]")
                .expect("program parses");

        let Stmt::Local {
            name,
            ty,
            dims,
            delocal_name,
            delocal_ty,
            delocal_dims,
            ..
        } = program.body.node
        else {
            panic!("expected local");
        };
        assert_eq!(name, "xs");
        assert_eq!(delocal_name, "xs");
        assert_eq!(dims, vec![Some(3)]);
        assert_eq!(delocal_dims, vec![Some(3)]);
        assert!(matches!(
            ty.expect("array annotation").node,
            TypeExpr::Array { .. }
        ));
        assert!(matches!(
            delocal_ty.expect("delocal array annotation").node,
            TypeExpr::Array { .. }
        ));
    }

    #[test]
    fn parses_type_first_delocal_refinement() {
        let program = parse_program("local int x = 0 x += 1 delocal int x where x > 0 = 1")
            .expect("program parses");

        let Stmt::Local {
            delocal_refinement, ..
        } = program.body.node
        else {
            panic!("expected local");
        };
        assert!(delocal_refinement.is_some());
    }

    #[test]
    fn parses_reverie_style_delocal_annotation_and_refinement() {
        let program = parse_program("local x: int = 0 x += 1 delocal x: int where x > 0 = 1")
            .expect("program parses");

        let Stmt::Local {
            delocal_ty,
            delocal_refinement,
            ..
        } = program.body.node
        else {
            panic!("expected local");
        };
        assert!(matches!(
            delocal_ty.expect("delocal type annotation").node,
            TypeExpr::Int { .. }
        ));
        assert!(delocal_refinement.is_some());
    }

    #[test]
    fn parses_global_declarations() {
        let program = parse_program("global x; global xs[3]; global bool flags[3]; x += 1")
            .expect("program parses");

        assert_eq!(program.globals.len(), 3);
        assert_eq!(program.globals[0].name, "x");
        assert_eq!(program.globals[0].len, 1);
        assert!(program.globals[0].dims.is_empty());
        assert_eq!(program.globals[1].name, "xs");
        assert_eq!(program.globals[1].len, 3);
        assert_eq!(program.globals[1].dims, [3]);
        assert_eq!(program.globals[2].name, "flags");
        assert_eq!(program.globals[2].len, 3);
        assert_eq!(program.globals[2].dims, [3]);
        assert!(matches!(
            program.globals[2]
                .ty
                .as_ref()
                .expect("bool annotation")
                .node,
            TypeExpr::Array { .. }
        ));
    }

    #[test]
    fn rejects_unsized_type_first_storage_declarations() {
        assert!(parse_program("global int xs[]; skip").is_err());
        assert!(parse_program("int xs[] = []").is_err());
        assert!(parse_program("procedure main() { int xs[] = [] }").is_err());
    }

    #[test]
    fn parses_legacy_bare_globals_before_procedures() {
        let program = parse_program(
            r#"
n x1 x2
list[12] matrix[3][2]

procedure main
  n += 1
"#,
        )
        .expect("program parses");

        assert_eq!(program.globals.len(), 5);
        assert_eq!(program.globals[0].name, "n");
        assert!(program.globals[0].dims.is_empty());
        assert_eq!(program.globals[1].name, "x1");
        assert_eq!(program.globals[2].name, "x2");
        assert_eq!(program.globals[3].name, "list");
        assert_eq!(program.globals[3].dims, [12]);
        assert_eq!(program.globals[4].name, "matrix");
        assert_eq!(program.globals[4].dims, [3, 2]);
        assert_eq!(program.procedures[0].name, "main");
    }

    #[test]
    fn keeps_top_level_statements_without_procedures() {
        let program = parse_program("x += 1").expect("program parses");

        assert!(program.globals.is_empty());
        assert!(matches!(program.body.node, Stmt::Update { .. }));
    }

    #[test]
    fn parses_janus_source_declarations_and_hoists_main_store() {
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

        assert_eq!(program.globals.len(), 3);
        assert_eq!(program.globals[0].name, "n");
        assert!(matches!(
            program.globals[0].ty.as_ref().expect("int annotation").node,
            TypeExpr::Int { .. }
        ));
        assert!(program.globals[0].init.is_some());
        assert_eq!(program.globals[1].name, "xs");
        assert_eq!(program.globals[1].len, 3);
        assert_eq!(program.globals[1].dims, [3]);
        assert!(matches!(
            program.globals[1]
                .init
                .as_ref()
                .expect("array initializer")
                .node,
            Expr::Array(_)
        ));
        assert_eq!(program.globals[2].name, "s");
        assert!(matches!(
            program.globals[2]
                .ty
                .as_ref()
                .expect("stack annotation")
                .node,
            TypeExpr::Stack
        ));

        let Stmt::Seq(statements) = &program.procedures[0].body.node else {
            panic!("expected declarations stripped from main body");
        };
        assert_eq!(statements.len(), 2);
        assert!(matches!(statements[0].node, Stmt::Assert { .. }));
    }

    #[test]
    fn uses_main_as_entry_when_body_is_omitted() {
        let program = parse_program("proc main() { skip }").expect("program parses");

        assert_eq!(program.procedures.len(), 1);
        assert!(matches!(
            program.body.node,
            Stmt::Call { ref name, ref args } if name == "main" && args.is_empty()
        ));
    }

    #[test]
    fn uses_legacy_main_fwd_as_entry_when_main_is_absent() {
        let program = parse_program("procedure main_fwd skip").expect("program parses");

        assert_eq!(program.procedures.len(), 1);
        assert!(matches!(
            program.body.node,
            Stmt::Call { ref name, ref args } if name == "main_fwd" && args.is_empty()
        ));
    }

    #[test]
    fn uses_legacy_test_entry_when_main_is_absent() {
        let program =
            parse_program("procedure helper skip procedure test skip").expect("program parses");

        assert_eq!(program.procedures.len(), 2);
        assert!(matches!(
            program.body.node,
            Stmt::Call { ref name, ref args } if name == "test" && args.is_empty()
        ));
    }

    #[test]
    fn parses_janus_procedure_keyword_and_parenless_calls() {
        let program =
            parse_program("procedure main { call step; uncall step } procedure step { skip }")
                .expect("program parses");

        assert_eq!(program.procedures.len(), 2);
        assert_eq!(program.procedures[0].name, "main");
        assert!(program.procedures[0].params.is_empty());
        let Stmt::Call { name, args } = &program.body.node else {
            panic!("expected implicit main call");
        };
        assert_eq!(name, "main");
        assert!(args.is_empty());
    }

    #[test]
    fn parses_place_procedure_arguments() {
        let program = parse_program("call test_rev(test_array[0])").expect("program parses");

        let Stmt::Call { name, args } = program.body.node else {
            panic!("expected call");
        };
        assert_eq!(name, "test_rev");
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].name, "test_array");
        assert_eq!(args[0].indices.len(), 1);
    }

    #[test]
    fn parses_unbraced_janus_procedures() {
        let program = parse_program(
            r#"
procedure step(int x)
  x += 1
  x -= 1

procedure main()
  call step(x)
"#,
        )
        .expect("program parses");

        assert_eq!(program.procedures.len(), 2);
        assert_eq!(program.procedures[0].name, "step");
        assert_eq!(program.procedures[1].name, "main");
        let Stmt::Seq(statements) = &program.procedures[0].body.node else {
            panic!("expected step sequence");
        };
        assert_eq!(statements.len(), 2);
    }

    #[test]
    fn parses_update() {
        let program = parse_program("x += y + 1").expect("program parses");
        let Stmt::Update { target, op, expr } = program.body.node else {
            panic!("expected update");
        };

        assert_eq!(target, Place::new("x".to_owned(), None));
        assert_eq!(op, UpdateOp::Add);
        assert!(matches!(
            expr.node,
            Expr::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn parses_increment_and_decrement_as_unit_updates() {
        let program = parse_program("x++; --xs[0]; ++y").expect("program parses");
        let Stmt::Seq(statements) = program.body.node else {
            panic!("expected sequence");
        };
        assert_eq!(statements.len(), 3);

        let Stmt::Update { target, op, expr } = &statements[0].node else {
            panic!("expected increment update");
        };
        assert_eq!(target, &Place::new("x".to_owned(), None));
        assert_eq!(*op, UpdateOp::Add);
        assert_eq!(
            expr.node,
            Expr::Int {
                value: 1,
                unit: None
            }
        );

        let Stmt::Update { target, op, expr } = &statements[1].node else {
            panic!("expected decrement update");
        };
        assert_eq!(target.name, "xs");
        assert_eq!(target.indices.len(), 1);
        assert_eq!(*op, UpdateOp::Sub);
        assert_eq!(
            expr.node,
            Expr::Int {
                value: 1,
                unit: None
            }
        );

        let Stmt::Update { target, op, expr } = &statements[2].node else {
            panic!("expected prefix increment update");
        };
        assert_eq!(target, &Place::new("y".to_owned(), None));
        assert_eq!(*op, UpdateOp::Add);
        assert_eq!(
            expr.node,
            Expr::Int {
                value: 1,
                unit: None
            }
        );
    }

    #[test]
    fn parses_swap() {
        let program = parse_program("x <=> y").expect("program parses");
        assert_eq!(
            program.body.node,
            Stmt::Swap {
                left: Place::new("x".to_owned(), None),
                right: Place::new("y".to_owned(), None)
            }
        );
    }

    #[test]
    fn parses_swap_call_alias() {
        let program = parse_program("swap(xs[0], ys[1])").expect("program parses");
        let Stmt::Swap { left, right } = program.body.node else {
            panic!("expected swap");
        };

        assert_eq!(left.name, "xs");
        assert_eq!(left.indices.len(), 1);
        assert_eq!(right.name, "ys");
        assert_eq!(right.indices.len(), 1);
    }

    #[test]
    fn parses_tape_io() {
        let program =
            parse_program("read x; write xs[0]; unread y; unwrite y").expect("program parses");
        let Stmt::Seq(statements) = program.body.node else {
            panic!("expected sequence");
        };

        assert!(matches!(statements[0].node, Stmt::Read { .. }));
        assert!(matches!(statements[1].node, Stmt::Write { .. }));
        assert!(matches!(statements[2].node, Stmt::Unread { .. }));
        assert!(matches!(statements[3].node, Stmt::Unwrite { .. }));
    }

    #[test]
    fn parses_array_literals_index_reads_and_index_updates() {
        let program =
            parse_program("local xs: array<int> = [1, 2, 3] xs[1] += xs[0] delocal xs = [1, 3, 3]")
                .expect("program parses");
        let Stmt::Local { ty, body, .. } = program.body.node else {
            panic!("expected local");
        };
        assert!(matches!(
            ty.expect("array annotation").node,
            TypeExpr::Array { .. }
        ));
        let Stmt::Update { target, expr, .. } = body.node else {
            panic!("expected indexed update");
        };
        assert_eq!(target.name, "xs");
        assert_eq!(target.indices.len(), 1);
        assert!(matches!(expr.node, Expr::Index { .. }));
    }

    #[test]
    fn parses_multidimensional_array_places_and_declarations() {
        let program = parse_program(
            r#"
procedure main()
  int matrix[3][2]
  matrix[1][0] += matrix[2][1]
"#,
        )
        .expect("program parses");

        assert_eq!(program.globals.len(), 1);
        assert_eq!(program.globals[0].name, "matrix");
        assert_eq!(program.globals[0].dims, [3, 2]);

        let Stmt::Update { target, expr, .. } = &program.procedures[0].body.node else {
            panic!("expected indexed update");
        };
        assert_eq!(target.indices.len(), 2);
        let Expr::Index { indices, .. } = &expr.node else {
            panic!("expected indexed read");
        };
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn parses_if_and_loop() {
        let source = r#"
if n != 0 then
  from i == 0 do
    a += b;
    a <=> b;
    i += 1
  loop
    skip
  until i == n
else
  skip
fi n != 0
"#;

        let program = parse_program(source).expect("program parses");
        assert!(matches!(program.body.node, Stmt::If { .. }));
    }

    #[test]
    fn parses_procedure_call_uncall_and_local() {
        let source = r#"
proc bump(x) {
  local t = 0
    t += x;
    x += 1
  delocal t = x - 1
}

call bump(n);
uncall bump(n)
"#;

        let program = parse_program(source).expect("program parses");
        assert_eq!(program.procedures.len(), 1);
        assert_eq!(program.procedures[0].name, "bump");
        assert_eq!(program.procedures[0].params[0].name, "x");
        assert!(matches!(program.body.node, Stmt::Seq(_)));
    }

    #[test]
    fn parses_unit_annotations_and_literals() {
        let source = r#"
proc pace(distance: int<m>, time: int<s>, speed: int<m/s>) {
  speed += distance / time
}

local distance: int<m> = 100<m>
  local time: int<s> = 10<s>
    local speed: int<m/s> = 0<m/s>
      call pace(distance, time, speed)
    delocal speed = 10<m/s>
  delocal time = 10<s>
delocal distance = 100<m>
"#;

        let program = parse_program(source).expect("program parses");
        assert_eq!(program.procedures[0].params[0].name, "distance");
        assert!(program.procedures[0].params[0].ty.is_some());
        assert!(matches!(program.body.node, Stmt::Local { .. }));
    }

    #[test]
    fn parses_unit_exponents() {
        let source = r#"
local acceleration: int<m/s^2> = 10<m/s^2>
  local time_sq: int<s^2> = 4<s^2>
    local distance: int<m> = 40<m>
      distance -= acceleration * time_sq
    delocal distance = 0<m>
  delocal time_sq = 4<s^2>
delocal acceleration = 10<m/s^2>
"#;

        let program = parse_program(source).expect("program parses");
        let Stmt::Local { ty, .. } = program.body.node else {
            panic!("expected local");
        };
        let TypeExpr::Int { unit: Some(unit) } = ty.expect("type annotation").node else {
            panic!("expected unitful int");
        };
        assert_eq!(unit.node.factors[0].name, "m");
        assert_eq!(unit.node.factors[0].exponent, 1);
        assert_eq!(unit.node.factors[1].name, "s");
        assert_eq!(unit.node.factors[1].exponent, -2);
    }

    #[test]
    fn parses_standalone_type_expr() {
        let ty = parse_type_expr("array<int<m/s^2>>").expect("type parses");
        assert!(matches!(ty.node, TypeExpr::Array { .. }));
    }

    #[test]
    fn parses_refinement_annotations() {
        let source = r#"
proc dec(n: int where n >= 0) {
  n -= 1
}

local n: int where n >= 0 = 3
  call dec(n)
delocal n = 2
"#;

        let program = parse_program(source).expect("program parses");
        assert!(program.procedures[0].params[0].refinement.is_some());
        let Stmt::Local { refinement, .. } = program.body.node else {
            panic!("expected local");
        };
        assert!(refinement.is_some());
    }

    #[test]
    fn rejects_unknown_syntax() {
        let diagnostics = parse_program("skop").expect_err("program is invalid");
        insta::assert_snapshot!(
            format!("{diagnostics:#?}"),
            @r###"
[
    SyntaxDiagnostic {
        message: "expected `!=`, `++`, `+=`, `--`, `-=`, `:`, `<=>`, `[` or `^=`, found end of input",
        span: 4..4,
        label: "the parser got stuck here",
    },
]
"###
        );
    }

    proptest! {
        #[test]
        fn parses_skip_with_optional_semicolon(trailing in proptest::prelude::any::<bool>()) {
            let source = if trailing { "skip;" } else { "skip" };
            prop_assert!(parse_program(source).is_ok());
        }
    }
}
