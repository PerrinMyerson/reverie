# Reverie for VS Code

Support for [Reverie](https://github.com/PerrinMyerson/reverie), a statically
typed reversible programming language.

- **Syntax highlighting** for `.rev` files.
- **Live diagnostics** as you type, from `reverie check --json`, including
  the checker's hints (for example where an update would destroy information
  and needs a witness).
- **Format Document** with `reverie fmt`.
- **Reverie: Show Inverse Program** opens the mechanically derived inverse
  beside the source.
- **Reverie: Run File** opens a terminal with `reverie run <file>` ready for
  `--var` seeds.

## Requirements

The `reverie` command on `PATH`:

```sh
pip install reverie-lang        # or, from a checkout: cargo install --path crates/reverie-cli
```

Set `reverie.path` if it lives elsewhere, and `reverie.legacyJanus` to check
legacy Janus sources.

## Installing from a checkout

```sh
cd editors/vscode
npx @vscode/vsce package        # produces reverie-lang-0.1.0.vsix
code --install-extension reverie-lang-0.1.0.vsix
```

Or symlink this folder into `~/.vscode/extensions/reverie-lang` and reload.
