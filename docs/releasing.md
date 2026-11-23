# Releasing

One tag publishes everything: Python wheels and sdist to PyPI, CLI binaries
and the VS Code extension (`.vsix`) to a GitHub release.

## One-time setup

1. On PyPI, create the `reverie-lang` project's **trusted publisher**:
   - owner `PerrinMyerson`, repository `reverie`;
   - workflow `release.yml`, environment `pypi`.

   No API token is stored anywhere.
2. In the GitHub repository settings, create an environment named `pypi`.
   Optionally require a reviewer, so each publish waits for approval.
3. Optional: to publish the extension to the VS Code Marketplace, create a
   `reverie-lang` publisher and run `npx @vscode/vsce publish` from
   `editors/vscode` with a personal access token. The release workflow only
   attaches the `.vsix` to the GitHub release.

## Each release

1. Bump the version in **both** places; the workflow refuses a tag that
   doesn't match them:
   - `Cargo.toml` (`[workspace.package] version`), which also sets the Python
     package version;
   - `editors/vscode/package.json` (`version`).
2. Add a section to `CHANGELOG.md`.
3. Commit and merge to `master`, then tag and push:

   ```sh
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. The `Release` workflow builds:
   - manylinux (x86_64, aarch64), macOS (x86_64, arm64) and Windows wheels,
     plus an sdist;
   - CLI binaries for Linux, macOS and Windows;
   - the VS Code extension.

   It publishes the Python packages to PyPI and creates a GitHub release with
   everything attached.

Run the workflow by hand (**Actions → Release → Run workflow**) to build all
artifacts without publishing.
