'use strict';

var fs = require('fs');
var spawnSync = require('child_process').spawnSync;
var _ = require('lodash');

var origReadme = fs.readFileSync('README.md', 'utf8').trim();
var lines = origReadme.split('\n').map(s => s + '\n');
var fencepos = lines.map((s, i) => [s, i]).filter(([s, i]) => s.indexOf('```') === 0);

var replacement = [];

var filesToContents = new Map();

_.chunk(fencepos, 2).forEach(([[_, i], [__, j]]) => {
  var language = lines[i].match(/```([^\s]+)/);
  language = language ? language[1] : language;

  var fname = null;
  if (lines[i + 1].indexOf('# export') === 0) { fname = lines[i + 1].match(/# export ([^\s]+)/)[1]; }
  var contentStart = i + 1 + (fname === null ? 0 : 1);
  var contents = lines.slice(contentStart, j).join('');

  if (language === 'py' || language === 'python') {
    contents = spawnSync('yapf', [], {input: contents, encoding: 'utf8'}).stdout;
    replacement.push({start: i, end: j, contentStart, contents});
  }

  if (fname) {
    if (filesToContents.has(fname)) {
      filesToContents.set(fname, filesToContents.get(fname) + contents);
    } else {
      if (language === 'py' || language === 'python') {
        // I need emoji!
        filesToContents.set(fname, '# -*- coding: utf-8 -*-\n\n' + contents)
      } else {
        filesToContents.set(fname, contents);
      }
    }
  }
});
for (const [fname, contents] of filesToContents) {
  const prev = fs.existsSync(fname) ? fs.readFileSync(fname, 'utf8').trim() : '___';
  if (prev !== contents.trim()) {
    console.log('Updating ' + fname);
    fs.writeFileSync(fname, contents);
  }
}

for (var ri = replacement.length - 1; ri >= 0; ri--) {
  var r = replacement[ri];
  for (var k = r.contentStart + 1; k < r.end; k++) { lines[k] = ''; }
  lines[r.contentStart] = r.contents;
}
const finalReadme = lines.join('').trim();
if (finalReadme !== origReadme) {
  console.log('Updating README.md');
  fs.writeFileSync('README.md', finalReadme);
}
