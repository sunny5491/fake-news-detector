// Pure-JS re-implementation of the scikit-learn TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
// + LogisticRegression baseline. Because the model is linear, each word's contribution to the
// decision is exact: contribution = tfidf_weight * coefficient. Positive pushes toward REAL.

const TOKEN = /[\p{L}\p{N}_]{2,}/gu; // sklearn default token_pattern (?u)\b\w\w+\b, after lowercasing

export function tokenize(text) {
  return (text.toLowerCase().match(TOKEN) || []);
}

export class TfidfLogReg {
  constructor(json) {
    this.b = json.b;
    this.idf = json.idf;
    this.coef = json.coef;
    this.index = new Map();
    json.vocab.forEach((t, i) => this.index.set(t, i));
  }

  // Returns { pReal, words: [[word, signedScore in -1..1], ...] }
  predict(text) {
    const toks = tokenize(text);
    const counts = new Map(); // feature index -> count
    const owners = new Map(); // feature index -> [token positions] (for attribution)
    const add = (term, pos) => {
      const i = this.index.get(term);
      if (i === undefined) return;
      counts.set(i, (counts.get(i) || 0) + 1);
      if (!owners.has(i)) owners.set(i, []);
      owners.get(i).push(...pos);
    };
    for (let k = 0; k < toks.length; k++) {
      add(toks[k], [k]);
      if (k + 1 < toks.length) add(toks[k] + " " + toks[k + 1], [k, k + 1]);
    }
    // sublinear tf, times idf, then l2 normalise
    const feats = [];
    let sq = 0;
    for (const [i, c] of counts) {
      const v = (1 + Math.log(c)) * this.idf[i];
      feats.push([i, v]);
      sq += v * v;
    }
    const norm = Math.sqrt(sq) || 1;
    let z = this.b;
    const perTok = new Float64Array(toks.length);
    for (const [i, v] of feats) {
      const contrib = (v / norm) * this.coef[i];
      z += contrib;
      const own = owners.get(i);
      const uniq = [...new Set(own)];
      for (const p of uniq) perTok[p] += contrib / uniq.length;
    }
    const pReal = 1 / (1 + Math.exp(-z));
    let m = 0;
    for (const s of perTok) m = Math.max(m, Math.abs(s));
    m = m || 1;
    const words = toks.map((w, k) => [w, Math.round((perTok[k] / m) * 1000) / 1000]);
    return { pReal, words };
  }
}
