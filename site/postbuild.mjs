// Runs after `notebooks build`. Pages live at src/<name>/index.html and are
// served at /<name>/; files in public/ are copied to the site root verbatim.
import { cpSync, readFileSync, writeFileSync } from "node:fs";

cpSync("public", "dist", { recursive: true });

// Home page: until it gets its own content, serve the design page at / too.
// The built page references assets as "../assets/ and url(../assets/; at the
// root those become ./assets/.
// When the home page gets its own content, delete this block and either add
// public/index.html or build src/index.html (list it in package.json's build).
const design = readFileSync("dist/design/index.html", "utf8");
writeFileSync("dist/index.html", design.replace(/(["(])\.\.\/assets\//g, "$1./assets/"));
