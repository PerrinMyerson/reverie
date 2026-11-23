// Run with: node test/diagnostics.test.js [path-to-reverie]
// Checks the extension's mapping from `reverie check --json` to diagnostics
// using a minimal stand-in for the `vscode` module.
const Module = require("module");
const assert = require("assert");
const cp = require("child_process");
const path = require("path");

class Position {
  constructor(line, character) { this.line = line; this.character = character; }
  isBefore(o) { return this.line < o.line || (this.line === o.line && this.character < o.character); }
}
class Range {
  constructor(a, b, c, d) {
    if (a instanceof Position) { this.start = a; this.end = b; }
    else { this.start = new Position(a, b); this.end = new Position(c, d); }
  }
}
const vscodeStub = {
  Position, Range,
  Diagnostic: class { constructor(range, message, severity) { Object.assign(this, { range, message, severity }); } },
  DiagnosticSeverity: { Error: 0 },
  DiagnosticRelatedInformation: class { constructor(location, message) { Object.assign(this, { location, message }); } },
  Location: class { constructor(uri, range) { Object.assign(this, { uri, range }); } },
  workspace: {}, window: {}, languages: {}, commands: {},
};
const resolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  return request === "vscode" ? "vscode" : resolve.call(this, request, ...rest);
};
require.cache.vscode = { id: "vscode", filename: "vscode", loaded: true, exports: vscodeStub };

const { toDiagnostics } = require("../extension.js");
const exe = process.argv[2] || "reverie";
const root = path.resolve(__dirname, "../../..");
const run = (file) =>
  JSON.parse(cp.spawnSync(exe, ["check", "--json", path.join(root, file)], { encoding: "utf8" }).stdout);

const bad = toDiagnostics(run("examples/irreversible_update.rev"), "uri");
assert.strictEqual(bad.length, 1);
assert.match(bad[0].message, /reads the value being changed/);
assert.deepStrictEqual([bad[0].range.start.line, bad[0].range.start.character], [0, 5]);
assert.ok(bad[0].relatedInformation.some((r) => /witness/.test(r.message)));
assert.strictEqual(toDiagnostics(run("examples/fib.rev"), "uri").length, 0);
console.log("ok: extension diagnostics mapping");
