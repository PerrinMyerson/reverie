// VS Code support for Reverie: live diagnostics from `reverie check --json`,
// formatting with `reverie fmt`, and commands to show the inverse program and
// run the current file. Plain JavaScript, no build step.

const vscode = require("vscode");
const cp = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

let diagnostics;
let timers = new Map();

function config() {
  const c = vscode.workspace.getConfiguration("reverie");
  return { exe: c.get("path", "reverie"), legacy: c.get("legacyJanus", false) };
}

// Commands run on a temporary copy so unsaved edits are checked too.
function withTempCopy(document, fn) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "reverie-"));
  const file = path.join(dir, path.basename(document.fileName || "untitled.rev"));
  fs.writeFileSync(file, document.getText());
  return fn(file).finally(() => fs.rmSync(dir, { recursive: true, force: true }));
}

function exec(args) {
  const { exe } = config();
  return new Promise((resolve) => {
    cp.execFile(exe, args, { maxBuffer: 64 * 1024 * 1024 }, (error, stdout, stderr) => {
      resolve({ code: error ? (typeof error.code === "number" ? error.code : -1) : 0, stdout, stderr, error });
    });
  });
}

function toRange(start, end) {
  const s = new vscode.Position(Math.max(start.line - 1, 0), Math.max(start.column - 1, 0));
  const e = end ? new vscode.Position(Math.max(end.line - 1, 0), Math.max(end.column - 1, 0)) : s;
  return new vscode.Range(s, e.isBefore(s) ? s : e);
}

// Convert one `reverie check --json` report into VS Code diagnostics.
function toDiagnostics(report, uri) {
  return (report.diagnostics || []).map((d) => {
    const range = d.start ? toRange(d.start, d.end) : new vscode.Range(0, 0, 0, 0);
    const diagnostic = new vscode.Diagnostic(range, d.message, vscode.DiagnosticSeverity.Error);
    diagnostic.source = "reverie";
    diagnostic.relatedInformation = (d.labels || [])
      .filter((l) => l.start)
      .map((l) => new vscode.DiagnosticRelatedInformation(new vscode.Location(uri, toRange(l.start, l.end)), l.message));
    return diagnostic;
  });
}

async function check(document) {
  if (document.languageId !== "reverie") return;
  const { legacy } = config();
  await withTempCopy(document, async (file) => {
    const args = ["check", "--json", ...(legacy ? ["--legacy-janus"] : []), file];
    const result = await exec(args);
    if (result.error && result.error.code === "ENOENT") {
      vscode.window.showWarningMessage(
        "Reverie: `reverie` was not found. Install it with `pip install reverie-lang` or set `reverie.path`."
      );
      return;
    }
    try {
      diagnostics.set(document.uri, toDiagnostics(JSON.parse(result.stdout), document.uri));
    } catch (e) {
      diagnostics.set(document.uri, [
        new vscode.Diagnostic(new vscode.Range(0, 0, 0, 0), (result.stderr || String(e)).trim(), vscode.DiagnosticSeverity.Error),
      ]);
    }
  });
}

function schedule(document) {
  clearTimeout(timers.get(document.uri.toString()));
  timers.set(document.uri.toString(), setTimeout(() => check(document), 300));
}

async function showInverse() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return;
  const { legacy } = config();
  await withTempCopy(editor.document, async (file) => {
    const result = await exec(["invert", ...(legacy ? ["--legacy-janus"] : []), file]);
    if (result.code !== 0) {
      vscode.window.showErrorMessage(`Reverie: ${result.stderr.trim() || "invert failed"}`);
      return;
    }
    const doc = await vscode.workspace.openTextDocument({ language: "reverie", content: result.stdout });
    vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
  });
}

function runFile() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return;
  const { exe } = config();
  const terminal = vscode.window.createTerminal("Reverie");
  terminal.show();
  // Left unsent so seeds can be added: --var name=value ...
  terminal.sendText(`${exe} run "${editor.document.fileName}" `, false);
}

const formatter = {
  async provideDocumentFormattingEdits(document) {
    const { legacy } = config();
    return withTempCopy(document, async (file) => {
      const result = await exec(["fmt", ...(legacy ? ["--legacy-janus"] : []), file]);
      if (result.code !== 0) return [];
      const whole = new vscode.Range(0, 0, document.lineCount, 0);
      return [vscode.TextEdit.replace(whole, result.stdout)];
    });
  },
};

function activate(context) {
  diagnostics = vscode.languages.createDiagnosticCollection("reverie");
  context.subscriptions.push(
    diagnostics,
    vscode.workspace.onDidOpenTextDocument(check),
    vscode.workspace.onDidSaveTextDocument(check),
    vscode.workspace.onDidChangeTextDocument((e) => schedule(e.document)),
    vscode.workspace.onDidCloseTextDocument((d) => diagnostics.delete(d.uri)),
    vscode.languages.registerDocumentFormattingEditProvider("reverie", formatter),
    vscode.commands.registerCommand("reverie.invert", showInverse),
    vscode.commands.registerCommand("reverie.run", runFile),
    vscode.commands.registerCommand("reverie.check", () => {
      const editor = vscode.window.activeTextEditor;
      if (editor) check(editor.document);
    })
  );
  vscode.workspace.textDocuments.forEach(check);
}

function deactivate() {}

module.exports = { activate, deactivate, toDiagnostics };
