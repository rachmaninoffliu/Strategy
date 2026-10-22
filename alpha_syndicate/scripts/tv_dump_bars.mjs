// Read-only dump of the FULL loaded bar series of a TradingView chart tab (pages history back first).
// usage: node tv_dump_bars.mjs <chartIdSubstring> <outJsonPath>
// Does not change symbol/resolution/studies. Needs TradingView started with --remote-debugging-port=9222.
import { createRequire } from 'module';
import { writeFileSync } from 'fs';

const require = createRequire('C:/Users/xinyu/tools/tradingview-mcp/package.json');
const CDP = require('chrome-remote-interface');

const [chartId, outPath] = process.argv.slice(2);
if (!chartId || !outPath) { console.error('usage: tv_dump_bars.mjs <chartId> <out.json>'); process.exit(2); }

const targets = await (await fetch('http://localhost:9222/json/list')).json();
const t = targets.find(x => x.type === 'page' && (x.url || '').includes('/chart/' + chartId));
if (!t) { console.error('chart tab not found:', chartId); process.exit(1); }

const c = await CDP({ host: 'localhost', port: 9222, target: t.id });
await c.Runtime.enable();
const expr = `(async () => {
  const w = window.TradingViewApi._activeChartWidgetWV.value();
  const ms = w._chartWidget.model().mainSeries();
  let stable = 0;
  for (let i = 0; i < 120; i++) {
    let more = true; try { more = ms.requestMoreDataAvailable(); } catch (e) {}
    const b = ms.bars(); const f0 = b.valueAt(b.firstIndex())[0];
    if (!more) break;
    try { ms.requestMoreData(2000); } catch (e) {}
    await new Promise(r => setTimeout(r, 1500));
    const b2 = ms.bars(); const f1 = b2.valueAt(b2.firstIndex())[0];
    if (f1 === f0) { stable++; if (stable >= 3) break; } else stable = 0;
  }
  const b = ms.bars(); const out = [];
  for (let i = b.firstIndex(); i <= b.lastIndex(); i++) { const v = b.valueAt(i); if (v) out.push([v[0], v[1], v[2], v[3], v[4], v[5]]); }
  return { symbol: ms.symbol(), interval: ms.interval(), n: out.length, bars: out };
})()`;
const res = await c.Runtime.evaluate({ expression: expr, awaitPromise: true, returnByValue: true, timeout: 180000 });
await c.close();
if (res.exceptionDetails) { console.error('page error', JSON.stringify(res.exceptionDetails).slice(0, 400)); process.exit(1); }
const v = res.result.value;
writeFileSync(outPath, JSON.stringify(v));
const first = new Date(v.bars[0][0] * 1000).toISOString().slice(0, 10), last = new Date(v.bars[v.bars.length - 1][0] * 1000).toISOString().slice(0, 10);
console.log(`${v.symbol} ${v.interval}: ${v.n} bars ${first} -> ${last}  -> ${outPath}`);
