// P2.87 card sketch, MADE-UP numbers. Two panels (left: rival a, right: rival b). x = 8 reuse tasks
// (e62/copy, e46/ident32, e53/lowercase, e54/elicit, c3/rank1, c3/copy, e58/rank1, e58/copy); y = share 0..1.2.
// Blue squares: fine-tunes (3 seeds); orange squares: random-subspace updates (3 seeds). Grey lines: y = 0 and y = 1.
const zlib = require("zlib"), fs = require("fs");
const W = 800, H = 360, img = Buffer.alloc(W * H * 3, 255);
function px(x, y, c) { if (x < 0 || y < 0 || x >= W || y >= H) return; const i = (y * W + x) * 3; img[i] = c[0]; img[i + 1] = c[1]; img[i + 2] = c[2]; }
function box(x, y, r, c) { for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) px(x + dx, y + dy, c); }
const ft = [1.09, 0.78, 0.76, 0.76, 0.84, 0.84, 0.86, 0.77];
const rivals = [[0.8, 0.7, 0.75, 0.7, 0.8, 0.82, 0.8, 0.7], [0.1, 0.2, 0.05, 0.15, 0.1, 0.2, 0.12, 0.18]];
const blue = [31, 119, 180], orange = [255, 127, 14], grey = [160, 160, 160], black = [0, 0, 0];
rivals.forEach((sub, p) => {
  const x0 = 40 + p * 400, x1 = x0 + 340, yv = v => Math.round(330 - v * 250);
  for (let x = x0; x <= x1; x++) { px(x, yv(0), black); px(x, yv(1), grey); }
  for (let y = yv(1.2); y <= yv(0); y++) px(x0, y, black);
  for (let t = 0; t < 8; t++) {
    const x = x0 + 25 + t * 40;
    [-0.02, 0, 0.02].forEach((j, s) => { box(x - 6, yv(ft[t] + j), 3, blue); box(x + 6, yv(sub[t] + j * (s + 1)), 3, orange); });
  }
});
function crc(b) { let c, t = []; for (let n = 0; n < 256; n++) { c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  let r = 0xffffffff; for (const x of b) r = t[(r ^ x) & 255] ^ (r >>> 8); return (r ^ 0xffffffff) >>> 0; }
function chunk(type, data) { const l = Buffer.alloc(4); l.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type), data]);
  const c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([l, td, c]); }
const raw = Buffer.alloc((W * 3 + 1) * H);
for (let y = 0; y < H; y++) { raw[y * (W * 3 + 1)] = 0; img.copy(raw, y * (W * 3 + 1) + 1, y * W * 3, (y + 1) * W * 3); }
const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr[8] = 8; ihdr[9] = 2;
fs.writeFileSync(process.argv[2], Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr),
  chunk("IDAT", zlib.deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]));
