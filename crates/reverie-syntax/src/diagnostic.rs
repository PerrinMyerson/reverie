use crate::SourceSpan;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxDiagnostic {
    pub message: String,
    pub span: SourceSpan,
    pub label: String,
}

impl SyntaxDiagnostic {
    pub fn new(message: impl Into<String>, span: SourceSpan, label: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span,
            label: label.into(),
        }
    }
}
