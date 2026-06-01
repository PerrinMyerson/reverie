use logos::Logos;

use crate::{SourceSpan, Spanned, SyntaxDiagnostic};

#[derive(Logos, Debug, Clone, PartialEq, Eq, Hash)]
#[logos(skip r"[ \t\r\n\f]+")]
#[logos(skip r"//[^\n]*")]
#[logos(skip r";[^\s\r\n][^\r\n]*")]
#[logos(skip r"/\*([^*]|\*[^/])*\*/")]
enum RawToken {
    #[token("skip")]
    Skip,
    #[token("assert")]
    Assert,
    #[token("if")]
    If,
    #[token("then")]
    Then,
    #[token("else")]
    Else,
    #[token("fi")]
    Fi,
    #[token("from")]
    From,
    #[token("do")]
    Do,
    #[token("loop")]
    Loop,
    #[token("until")]
    Until,
    #[token("iterate")]
    Iterate,
    #[token("to")]
    To,
    #[token("by")]
    By,
    #[token("end")]
    End,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("proc")]
    Proc,
    #[token("procedure")]
    Procedure,
    #[token("global")]
    Global,
    #[token("call")]
    Call,
    #[token("uncall")]
    Uncall,
    #[token("read")]
    Read,
    #[token("unread")]
    Unread,
    #[token("write")]
    Write,
    #[token("unwrite")]
    Unwrite,
    #[token("local")]
    Local,
    #[token("delocal")]
    Delocal,
    #[token("int")]
    IntType,
    #[token("bool")]
    BoolType,
    #[token("array")]
    ArrayType,
    #[token("stack")]
    StackType,
    #[token("where")]
    Where,
    #[token("size")]
    Size,
    #[token("empty")]
    Empty,
    #[token("top")]
    Top,
    #[token("nil")]
    Nil,
    #[token("push")]
    Push,
    #[token("pop")]
    Pop,
    #[token("show")]
    Show,
    #[token("printf")]
    Printf,

    #[token(";")]
    Semicolon,
    #[token(":")]
    Colon,
    #[token(",")]
    Comma,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("=")]
    Eq,

    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("++")]
    PlusPlus,
    #[token("--")]
    MinusMinus,
    #[token("^=")]
    XorEq,
    #[token("<=>")]
    Swap,

    #[token("==")]
    EqEq,
    #[token("!=")]
    BangEq,
    #[token("!")]
    Bang,
    #[token("#")]
    Hash,
    #[token("~")]
    Tilde,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GtEq,
    #[token("&&")]
    AndAnd,
    #[token("&")]
    Amp,
    #[token("||")]
    OrOr,
    #[token("|")]
    Pipe,

    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*/")]
    StarSlash,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("\\")]
    Backslash,
    #[token("^")]
    Caret,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,

    #[regex(r"[0-9]+")]
    Int,
    #[regex(r#""([^"\\\n\r]|\\.)*""#)]
    String,
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*")]
    Ident,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Token {
    Skip,
    Assert,
    If,
    Then,
    Else,
    Fi,
    From,
    Do,
    Loop,
    Until,
    Iterate,
    To,
    By,
    End,
    True,
    False,
    Proc,
    Global,
    Call,
    Uncall,
    Read,
    Unread,
    Write,
    Unwrite,
    Local,
    Delocal,
    IntType,
    BoolType,
    ArrayType,
    StackType,
    Where,
    Size,
    Empty,
    Top,
    Nil,
    Push,
    Pop,
    Show,
    Printf,
    Semicolon,
    Colon,
    Comma,
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Eq,
    PlusEq,
    MinusEq,
    PlusPlus,
    MinusMinus,
    XorEq,
    Swap,
    EqEq,
    BangEq,
    Bang,
    LtEq,
    GtEq,
    Amp,
    Pipe,
    AndAnd,
    OrOr,
    Plus,
    Minus,
    StarSlash,
    Star,
    Slash,
    Percent,
    Caret,
    Lt,
    Gt,
    Int(i64),
    String(String),
    Ident(String),
}

impl Token {
    pub fn display(&self) -> String {
        match self {
            Self::Skip => "`skip`".to_owned(),
            Self::Assert => "`assert`".to_owned(),
            Self::If => "`if`".to_owned(),
            Self::Then => "`then`".to_owned(),
            Self::Else => "`else`".to_owned(),
            Self::Fi => "`fi`".to_owned(),
            Self::From => "`from`".to_owned(),
            Self::Do => "`do`".to_owned(),
            Self::Loop => "`loop`".to_owned(),
            Self::Until => "`until`".to_owned(),
            Self::Iterate => "`iterate`".to_owned(),
            Self::To => "`to`".to_owned(),
            Self::By => "`by`".to_owned(),
            Self::End => "`end`".to_owned(),
            Self::True => "`true`".to_owned(),
            Self::False => "`false`".to_owned(),
            Self::Proc => "`proc`".to_owned(),
            Self::Global => "`global`".to_owned(),
            Self::Call => "`call`".to_owned(),
            Self::Uncall => "`uncall`".to_owned(),
            Self::Read => "`read`".to_owned(),
            Self::Unread => "`unread`".to_owned(),
            Self::Write => "`write`".to_owned(),
            Self::Unwrite => "`unwrite`".to_owned(),
            Self::Local => "`local`".to_owned(),
            Self::Delocal => "`delocal`".to_owned(),
            Self::IntType => "`int`".to_owned(),
            Self::BoolType => "`bool`".to_owned(),
            Self::ArrayType => "`array`".to_owned(),
            Self::StackType => "`stack`".to_owned(),
            Self::Where => "`where`".to_owned(),
            Self::Size => "`size`".to_owned(),
            Self::Empty => "`empty`".to_owned(),
            Self::Top => "`top`".to_owned(),
            Self::Nil => "`nil`".to_owned(),
            Self::Push => "`push`".to_owned(),
            Self::Pop => "`pop`".to_owned(),
            Self::Show => "`show`".to_owned(),
            Self::Printf => "`printf`".to_owned(),
            Self::Semicolon => "`;`".to_owned(),
            Self::Colon => "`:`".to_owned(),
            Self::Comma => "`,`".to_owned(),
            Self::LBrace => "`{`".to_owned(),
            Self::RBrace => "`}`".to_owned(),
            Self::LParen => "`(`".to_owned(),
            Self::RParen => "`)`".to_owned(),
            Self::LBracket => "`[`".to_owned(),
            Self::RBracket => "`]`".to_owned(),
            Self::Eq => "`=`".to_owned(),
            Self::PlusEq => "`+=`".to_owned(),
            Self::MinusEq => "`-=`".to_owned(),
            Self::PlusPlus => "`++`".to_owned(),
            Self::MinusMinus => "`--`".to_owned(),
            Self::XorEq => "`^=`".to_owned(),
            Self::Swap => "`<=>`".to_owned(),
            Self::EqEq => "`==`".to_owned(),
            Self::BangEq => "`!=`".to_owned(),
            Self::Bang => "`!`".to_owned(),
            Self::LtEq => "`<=`".to_owned(),
            Self::GtEq => "`>=`".to_owned(),
            Self::Amp => "`&`".to_owned(),
            Self::Pipe => "`|`".to_owned(),
            Self::AndAnd => "`&&`".to_owned(),
            Self::OrOr => "`||`".to_owned(),
            Self::Plus => "`+`".to_owned(),
            Self::Minus => "`-`".to_owned(),
            Self::StarSlash => "`*/`".to_owned(),
            Self::Star => "`*`".to_owned(),
            Self::Slash => "`/`".to_owned(),
            Self::Percent => "`%`".to_owned(),
            Self::Caret => "`^`".to_owned(),
            Self::Lt => "`<`".to_owned(),
            Self::Gt => "`>`".to_owned(),
            Self::Int(value) => format!("integer `{value}`"),
            Self::String(_) => "string literal".to_owned(),
            Self::Ident(name) => format!("identifier `{name}`"),
        }
    }
}

pub fn lex(source: &str) -> Result<Vec<Spanned<Token>>, Vec<SyntaxDiagnostic>> {
    let mut lexer = RawToken::lexer(source);
    let mut tokens = Vec::new();
    let mut diagnostics = Vec::new();

    while let Some(result) = lexer.next() {
        let span = lexer.span();
        match result {
            Ok(raw) => match token_from_raw(raw, lexer.slice(), span.clone()) {
                Ok(token) => tokens.push(Spanned::new(token, span)),
                Err(diagnostic) => diagnostics.push(diagnostic),
            },
            Err(()) => push_unrecognized_token(&mut diagnostics, span),
        }
    }

    if diagnostics.is_empty() {
        Ok(tokens)
    } else {
        Err(diagnostics)
    }
}

fn token_from_raw(raw: RawToken, slice: &str, span: SourceSpan) -> Result<Token, SyntaxDiagnostic> {
    Ok(match raw {
        RawToken::Skip => Token::Skip,
        RawToken::Assert => Token::Assert,
        RawToken::If => Token::If,
        RawToken::Then => Token::Then,
        RawToken::Else => Token::Else,
        RawToken::Fi => Token::Fi,
        RawToken::From => Token::From,
        RawToken::Do => Token::Do,
        RawToken::Loop => Token::Loop,
        RawToken::Until => Token::Until,
        RawToken::Iterate => Token::Iterate,
        RawToken::To => Token::To,
        RawToken::By => Token::By,
        RawToken::End => Token::End,
        RawToken::True => Token::True,
        RawToken::False => Token::False,
        RawToken::Proc | RawToken::Procedure => Token::Proc,
        RawToken::Global => Token::Global,
        RawToken::Call => Token::Call,
        RawToken::Uncall => Token::Uncall,
        RawToken::Read => Token::Read,
        RawToken::Unread => Token::Unread,
        RawToken::Write => Token::Write,
        RawToken::Unwrite => Token::Unwrite,
        RawToken::Local => Token::Local,
        RawToken::Delocal => Token::Delocal,
        RawToken::IntType => Token::IntType,
        RawToken::BoolType => Token::BoolType,
        RawToken::ArrayType => Token::ArrayType,
        RawToken::StackType => Token::StackType,
        RawToken::Where => Token::Where,
        RawToken::Size => Token::Size,
        RawToken::Empty => Token::Empty,
        RawToken::Top => Token::Top,
        RawToken::Nil => Token::Nil,
        RawToken::Push => Token::Push,
        RawToken::Pop => Token::Pop,
        RawToken::Show => Token::Show,
        RawToken::Printf => Token::Printf,
        RawToken::Semicolon => Token::Semicolon,
        RawToken::Colon => Token::Colon,
        RawToken::Comma => Token::Comma,
        RawToken::LBrace => Token::LBrace,
        RawToken::RBrace => Token::RBrace,
        RawToken::LParen => Token::LParen,
        RawToken::RParen => Token::RParen,
        RawToken::LBracket => Token::LBracket,
        RawToken::RBracket => Token::RBracket,
        RawToken::Eq => Token::Eq,
        RawToken::PlusEq => Token::PlusEq,
        RawToken::MinusEq => Token::MinusEq,
        RawToken::PlusPlus => Token::PlusPlus,
        RawToken::MinusMinus => Token::MinusMinus,
        RawToken::XorEq => Token::XorEq,
        RawToken::Swap => Token::Swap,
        RawToken::EqEq => Token::EqEq,
        RawToken::BangEq => Token::BangEq,
        RawToken::Bang => Token::Bang,
        RawToken::Hash => Token::BangEq,
        RawToken::Tilde => Token::Bang,
        RawToken::LtEq => Token::LtEq,
        RawToken::GtEq => Token::GtEq,
        RawToken::AndAnd => Token::AndAnd,
        RawToken::Amp => Token::Amp,
        RawToken::OrOr => Token::OrOr,
        RawToken::Pipe => Token::Pipe,
        RawToken::Plus => Token::Plus,
        RawToken::Minus => Token::Minus,
        RawToken::StarSlash => Token::StarSlash,
        RawToken::Star => Token::Star,
        RawToken::Slash => Token::Slash,
        RawToken::Percent => Token::Percent,
        RawToken::Backslash => Token::Percent,
        RawToken::Caret => Token::Caret,
        RawToken::Lt => Token::Lt,
        RawToken::Gt => Token::Gt,
        RawToken::Ident => keyword_token(slice).unwrap_or_else(|| Token::Ident(slice.to_owned())),
        RawToken::String => Token::String(decode_string(slice, span)?),
        RawToken::Int => Token::Int(slice.parse().map_err(|_| {
            SyntaxDiagnostic::new(
                "integer literal is out of range",
                span,
                "Phase 1 integers must fit in signed i64",
            )
        })?),
    })
}

fn keyword_token(slice: &str) -> Option<Token> {
    Some(match slice.to_ascii_lowercase().as_str() {
        "skip" => Token::Skip,
        "assert" => Token::Assert,
        "if" => Token::If,
        "then" => Token::Then,
        "else" => Token::Else,
        "fi" => Token::Fi,
        "from" => Token::From,
        "do" => Token::Do,
        "loop" => Token::Loop,
        "until" => Token::Until,
        "iterate" => Token::Iterate,
        "to" => Token::To,
        "by" => Token::By,
        "end" => Token::End,
        "true" => Token::True,
        "false" => Token::False,
        "proc" | "procedure" => Token::Proc,
        "global" => Token::Global,
        "call" => Token::Call,
        "uncall" => Token::Uncall,
        "read" => Token::Read,
        "unread" => Token::Unread,
        "write" => Token::Write,
        "unwrite" => Token::Unwrite,
        "local" => Token::Local,
        "delocal" => Token::Delocal,
        "int" => Token::IntType,
        "bool" => Token::BoolType,
        "array" => Token::ArrayType,
        "stack" => Token::StackType,
        "where" => Token::Where,
        "size" => Token::Size,
        "empty" => Token::Empty,
        "top" => Token::Top,
        "nil" => Token::Nil,
        "push" => Token::Push,
        "pop" => Token::Pop,
        "show" => Token::Show,
        "printf" => Token::Printf,
        _ => return None,
    })
}

fn decode_string(slice: &str, span: SourceSpan) -> Result<String, SyntaxDiagnostic> {
    let inner = slice
        .strip_prefix('"')
        .and_then(|slice| slice.strip_suffix('"'))
        .expect("logos string token includes quotes");
    let mut decoded = String::new();
    let mut chars = inner.chars();

    while let Some(ch) = chars.next() {
        if ch != '\\' {
            decoded.push(ch);
            continue;
        }

        let Some(escaped) = chars.next() else {
            return Err(SyntaxDiagnostic::new(
                "invalid string escape",
                span,
                "string escape is missing a character",
            ));
        };

        match escaped {
            'n' => decoded.push('\n'),
            'r' => decoded.push('\r'),
            't' => decoded.push('\t'),
            '"' => decoded.push('"'),
            '\\' => decoded.push('\\'),
            '0' => decoded.push('\0'),
            other => {
                return Err(SyntaxDiagnostic::new(
                    format!("unsupported string escape `\\{other}`"),
                    span,
                    "supported escapes are \\n, \\r, \\t, \\\", \\\\, and \\0",
                ));
            }
        }
    }

    Ok(decoded)
}

fn push_unrecognized_token(diagnostics: &mut Vec<SyntaxDiagnostic>, span: SourceSpan) {
    if let Some(last) = diagnostics.last_mut()
        && last.message == "unrecognized token"
        && last.span.end == span.start
    {
        last.span.end = span.end;
        return;
    }

    diagnostics.push(SyntaxDiagnostic::new(
        "unrecognized token",
        span,
        "this token is not valid Reverie syntax",
    ));
}
