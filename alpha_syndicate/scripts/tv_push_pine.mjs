// Safe push of a local .pine file into the TradingView Pine editor over CDP.
// Refuses to write unless the editor is bound to the expected script name (overwrite-hazard guard).
// usage: node push_pine.mjs <file.pine> "<expected editor title>"
import { createRequire } from 'module';
import { readFileSync } from 'fs';

const require = createRequire('C:/Users/xinyu/tools/tradingview-mcp/package.json');
const CDP = require('chrome-remote-interface');

const [file, expected] = process.argv.slice(2);
if (!file || !expected) { console.error('usage: push_pine.mjs <file> <expected title>'); process.exit(2); }
const src = readFileSync(file, 'utf-8');

const withEditor = (body) => `(function(){
  var c=document.querySelector(".monaco-editor.pine-editor-monaco"); if(!c) return {err:"no pine editor"};
  var el=c,fk; for(var i=0;i<20;i++){ if(!el)break; fk=Object.keys(el).find(function(k){return k.startsWith("__reactFiber$")}); if(fk)break; el=el.parentElement }
  if(!fk) return {err:"no react fiber"};
  var cur=el[fk], ed=null;
  for(var d=0;d<15;d++){ if(!cur)break;
    if(cur.memoizedProps&&cur.memoizedProps.value&&cur.memoizedProps.value.monacoEnv){
      var env=cur.memoizedProps.value.monacoEnv;
      if(env.editor&&typeof env.editor.getEditors==="function"){ var eds=env.editor.getEditors(); if(eds.length>0){ ed=eds[0]; break } } }
    cur=cur.return }
  if(!ed) return {err:"no monaco editor"};
  ${body}
})()`;

const targets = await (await fetch('http://localhost:9222/json/list')).json();
let done = false;
const timeout = (p, ms, what) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout: ' + what)), ms))]);
for (const t of targets.filter(t => t.type === 'page' && (t.url || '').includes('tradingview.com/chart'))) {
  const c = await timeout(CDP({ host: 'localhost', port: 9222, target: t.id }), 10000, 'connect');
  try {
    await timeout(c.Runtime.enable(), 10000, 'enable');
    const probe = (await timeout(c.Runtime.evaluate({ expression: withEditor('return {ok:true};'), returnByValue: true }), 15000, 'probe')).result?.value;
    if (!probe || probe.err) continue;
    const titles = (await c.Runtime.evaluate({ expression: `[...document.querySelectorAll('h2')].map(h=>h.textContent.trim())`, returnByValue: true })).result?.value || [];
    console.log('editor title(s):', JSON.stringify(titles));
    if (titles.length !== 1 || titles[0] !== expected) {
      console.error(`ABORT: editor is bound to ${JSON.stringify(titles)}, expected "${expected}". Nothing written.`);
      process.exit(3);
    }
    const res = (await c.Runtime.evaluate({
      expression: withEditor(`ed.setValue(${JSON.stringify(src)}); var m=ed.getModel(); return {ok:true, lines:m.getLineCount(), chars:m.getValueLength()};`),
      returnByValue: true,
    })).result?.value;
    console.log('pushed:', JSON.stringify(res), '| file lines:', src.split('\n').length);
    done = true;
    break;
  } finally { await c.close(); }
}
if (!done) { console.error('No TradingView page with a Pine editor found'); process.exit(1); }
