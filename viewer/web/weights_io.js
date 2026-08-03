// Decode the fp16 binaries written by export_weights.py into Float32Arrays.

export function decodeFp16(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * m * 2 ** -24
           : e === 31 ? (m ? NaN : s * Infinity)
           : s * (1024 + m) * 2 ** (e - 25);
  }
  return out;
}

// index: {name: {offset (bytes), shape}} from the manifest; buffer: the .bin ArrayBuffer
export function tensors(index, buffer) {
  const out = {};
  for (const [name, { offset, shape }] of Object.entries(index)) {
    const n = shape.reduce((a, b) => a * b, 1);
    out[name] = decodeFp16(new Uint16Array(buffer, offset, n));
  }
  return out;
}
