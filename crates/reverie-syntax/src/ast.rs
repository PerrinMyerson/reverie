use std::ops::Range;

pub type SourceSpan = Range<usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned<T> {
    pub node: T,
    pub span: SourceSpan,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: SourceSpan) -> Self {
        Self { node, span }
    }
}

pub type SpannedStmt = Spanned<Stmt>;
pub type SpannedExpr = Spanned<Expr>;
pub type SpannedUnit = Spanned<UnitExpr>;
pub type SpannedType = Spanned<TypeExpr>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub globals: Vec<GlobalDecl>,
    pub procedures: Vec<Proc>,
    pub body: SpannedStmt,
}

impl Program {
    pub fn new(body: SpannedStmt) -> Self {
        Self {
            globals: Vec::new(),
            procedures: Vec::new(),
            body,
        }
    }

    pub fn with_procedures(procedures: Vec<Proc>, body: SpannedStmt) -> Self {
        Self {
            globals: Vec::new(),
            procedures,
            body,
        }
    }

    pub fn with_globals_and_procedures(
        globals: Vec<GlobalDecl>,
        procedures: Vec<Proc>,
        body: SpannedStmt,
    ) -> Self {
        Self {
            globals,
            procedures,
            body,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalDecl {
    pub name: String,
    pub len: usize,
    pub dims: Vec<usize>,
    pub ty: Option<SpannedType>,
    pub init: Option<SpannedExpr>,
    pub span: SourceSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proc {
    pub name: String,
    pub params: Vec<Param>,
    pub body: SpannedStmt,
    pub span: SourceSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub ty: Option<SpannedType>,
    pub refinement: Option<SpannedExpr>,
    pub span: SourceSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeExpr {
    Int {
        unit: Option<SpannedUnit>,
    },
    Bool,
    Array {
        element: Box<SpannedType>,
    },
    Tensor {
        element: Box<SpannedType>,
        shape: Vec<usize>,
    },
    Witness {
        inner: Box<SpannedType>,
    },
    Stack,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Skip,
    Seq(Vec<SpannedStmt>),
    Assert {
        condition: SpannedExpr,
    },
    Update {
        target: Place,
        op: UpdateOp,
        expr: SpannedExpr,
    },
    Swap {
        left: Place,
        right: Place,
    },
    Push {
        source: Place,
        stack: String,
    },
    Pop {
        target: Place,
        stack: String,
    },
    If {
        entry: SpannedExpr,
        then_branch: Box<SpannedStmt>,
        else_branch: Box<SpannedStmt>,
        exit: SpannedExpr,
    },
    Loop {
        entry: SpannedExpr,
        body: Box<SpannedStmt>,
        step: Box<SpannedStmt>,
        exit: SpannedExpr,
    },
    Iterate {
        name: String,
        start: SpannedExpr,
        step: SpannedExpr,
        end: SpannedExpr,
        body: Box<SpannedStmt>,
    },
    Call {
        name: String,
        args: Vec<Place>,
    },
    Uncall {
        name: String,
        args: Vec<Place>,
    },
    Read {
        target: Place,
    },
    Unread {
        target: Place,
    },
    Write {
        source: Place,
    },
    Unwrite {
        source: Place,
    },
    Show {
        targets: Vec<String>,
    },
    Printf {
        format: String,
        args: Vec<SpannedExpr>,
    },
    Local {
        name: String,
        ty: Option<SpannedType>,
        dims: Vec<Option<usize>>,
        refinement: Option<SpannedExpr>,
        init: SpannedExpr,
        body: Box<SpannedStmt>,
        delocal_name: String,
        delocal_ty: Option<SpannedType>,
        delocal_dims: Vec<Option<usize>>,
        delocal_refinement: Option<SpannedExpr>,
        delocal: SpannedExpr,
    },
    Declare {
        name: String,
        ty: SpannedType,
        len: usize,
        dims: Vec<usize>,
        init: Option<SpannedExpr>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Place {
    pub name: String,
    pub indices: Vec<SpannedExpr>,
}

impl Place {
    pub fn new(name: String, index: Option<SpannedExpr>) -> Self {
        Self {
            name,
            indices: index.into_iter().collect(),
        }
    }

    pub fn with_indices(name: String, indices: Vec<SpannedExpr>) -> Self {
        Self { name, indices }
    }

    pub fn is_indexed(&self) -> bool {
        !self.indices.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateOp {
    Add,
    Sub,
    Xor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Int {
        value: i64,
        unit: Option<SpannedUnit>,
    },
    Bool(bool),
    Array(Vec<SpannedExpr>),
    Nil,
    Var(String),
    Index {
        target: String,
        indices: Vec<SpannedExpr>,
    },
    Empty {
        target: String,
    },
    Top {
        target: String,
    },
    Size {
        target: String,
    },
    Unary {
        op: UnaryOp,
        expr: Box<SpannedExpr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<SpannedExpr>,
        right: Box<SpannedExpr>,
    },
    Call {
        name: String,
        args: Vec<SpannedExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    FixedMul,
    Div,
    Rem,
    Shl,
    Shr,
    BitAnd,
    BitXor,
    BitOr,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitExpr {
    pub factors: Vec<UnitFactor>,
}

impl UnitExpr {
    pub fn new(factors: Vec<UnitFactor>) -> Self {
        Self { factors }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitFactor {
    pub name: String,
    pub exponent: i32,
}
