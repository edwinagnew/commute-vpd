// P2.90 card sketches, MADE-UP numbers. argv[2] = out PNG, argv[3] = "67m" or "qwen". Left panel rival a, right rival b.
// 67m: x = old share (known3.json), y = new share, 48 states; grey = identity line. qwen: x = 6 sides; blue = fine-tune
// seeds' shares, orange = random reference; y = share 0..1.2, grey line y = 1.
const zlib = require("zlib"), fs = require("fs");
const W = 800, H = 360, img = Buffer.alloc(W * H * 3, 255);
function px(x, y, c) { if (x < 0 || y < 0 || x >= W || y >= H) return; const i = (y * W + x) * 3; img[i] = c[0]; img[i + 1] = c[1]; img[i + 2] = c[2]; }
function box(x, y, r, c) { for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) px(x + dx, y + dy, c); }
const blue = [31, 119, 180], orange = [255, 127, 14], grey = [160, 160, 160], black = [0, 0, 0];
const mode = process.argv[3];
for (let p = 0; p < 2; p++) {
  const x0 = 40 + p * 400, yv = v => Math.round(330 - v * 250), xv = v => Math.round(x0 + v * 280);
  for (let x = x0; x <= x0 + 340; x++) px(x, yv(0), black);
  for (let y = yv(1.2); y <= yv(0); y++) px(x0, y, black);
  if (mode === "67m") {
    for (let v = 0; v <= 1.2; v += 0.002) px(xv(v), yv(v), grey);
    for (let s = 0; s < 48; s++) { const o = 0.2 + 0.8 * ((s * 37) % 48) / 48; box(xv(o), yv(o + (p === 1 && s % 9 === 0 ? 0.15 : 0)), 3, blue); }
  } else {
    for (let x = x0; x <= x0 + 340; x++) px(x, yv(1), grey);
    const ft = [0.85, 0.9, 0.8, 0.75, 0.95, 0.7], rnd = p === 0 ? [0.3, 0.2, 0.35, 0.25, 0.4, 0.3] : [0.85, 0.88, 0.82, 0.8, 0.9, 0.75];
    for (let t = 0; t < 6; t++) { const x = x0 + 30 + t * 50; [-0.03, 0, 0.03].forEach(j => box(x - 6, yv(ft[t] + j), 3, blue)); box(x + 8, yv(rnd[t]), 4, orange); }
  }
}
function crc(b) { let c, t = []; for (let n = 0; n < 256; n++) { c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  let r = 0xffffffff; for (const x of b) r = t[(r ^ x) & 255] ^ (r >>> 8); return (r ^ 0xffffffff) >>> 0; }
function chunk(type, data) { const l = Buffer.alloc(4); l.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type), data]);
  const c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([l, td, c]); }
const raw = Buffer.alloc((W * 3 + 1) * H);
for (let y = 0; y < H; y++) { raw[y * (W * 3 + 1)] = 0; img.copy(raw, y * (W * 3 + 1) + 1, y * W * 3, (y + 1) * W * 3); }
const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr[8] = 8; ihdr[9] = 2;
fs.writeFileSync(process.argv[2], Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr),
  chunk("IDAT", zlib.deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]));
