#!/usr/bin/env node
// CLI entry for `npx livesub` / a global `livesub`. Thin wrapper so the shebang
// is guaranteed on the bin file; the real server is the compiled dist/server.js
// (importing it starts the HTTP + WS listener). Config (.env, model dir) is read
// from the current working directory — see src/config.ts.
import "../dist/server.js";
