const templates = require("../app/static/templates.js");
const fs = require("fs");
const path = require("path");

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

const items = templates.templates;
assert(items.length === 17, "seventeen templates are listed");
const cmds = items.map((item) => item.cmd);
assert(new Set(cmds).size === cmds.length, "each template command is unique");
const files = items.map((item) => item.file);
assert(new Set(files).size === files.length, "each template file is unique");
items.forEach((item) => {
  const file = path.join(__dirname, "../samples", item.file);
  assert(fs.existsSync(file), item.file + " is a sample program");
  assert(item.title && item.expect, item.cmd + " has a title and a result");
});
const byCmd = Object.fromEntries(items.map((item) => [item.cmd, item]));
assert(byCmd.lineTemplate.expect === "合格 3、不合格 2", "the blob line stays 3 and 2");
assert(byCmd.lineHolesTemplate.expect === "合格 2、不合格 3", "the hole line stays 2 and 3");
assert(byCmd.holesTemplate.expect.includes("直徑約 28"), "the hole template names the diameter");
assert(byCmd.charsTemplate.expect.includes("VISDEV 168"), "the character template names the word");
assert(byCmd.colorTemplate.expect.includes("1200"), "the color template names the area");
assert(byCmd.missingTemplate.expect.includes("8"), "the missing-part template names eight parts");
assert(byCmd.fitTemplate.expect.includes("半徑 20"), "the fit template names the circle radius");
const page = fs.readFileSync(path.join(__dirname, "../app/static/index.html"), "utf8");
assert(page.includes('id="templateFly"'), "the template menu is filled from the catalog");
assert(page.includes('data-cmd="templateList"'), "the file menu opens the catalog");
assert(!page.includes('data-cmd="holesTemplate"'), "template buttons are not hardcoded beside the catalog");

console.log("templates ok");
