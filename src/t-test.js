// Borrowed from https://gist.github.com/leeoniya/747e00a58cd5980d1a9c8d8e8979c68a

// Welch t-test:  ported from C# code
// - https://msdn.microsoft.com/ja-jp/magazine/mt620016.aspx

// ACM Algorithm #209 Gauss
const P0 = [
    +0.000124818987,
    -0.001075204047,
    +0.005198775019,
    -0.019198292004,
    +0.059054035642,
    -0.151968751364,
    +0.319152932694,
    -0.531923007300,
    +0.797884560593,
];
const P1 = [
    -0.000045255659,
    +0.000152529290,
    -0.000019538132,
    -0.000676904986,
    +0.001390604284,
    -0.000794620820,
    -0.002034254874,
    +0.006549791214,
    -0.010557625006,
    +0.011630447319,
    -0.009279453341,
    +0.005353579108,
    -0.002141268741,
    +0.000535310849,
    +0.999936657524,
];
function pol(x, c) {
    return c.reduce((r, c) => r * x + c, 0);
}

// integral amount of ND(mean=0, SD=1) returns between 0 and 1
// - gauss(z) = (1 + erf(z * 2**-0.5))/2
function gauss(z) {
    if (z === 0) return 0.5;
    const y = Math.abs(z) / 2;
    const p = y >= 3 ? 1 : y < 1 ? pol(y * y, P0) * y * 2 : pol(y - 2, P1);
    return z > 0 ? (1 + p) / 2 : (1 - p) / 2;
}

// ACM Algorithm #395 (student t-distribution: df as double)
// t-dist : distribution for average of (df+1)-samples from ND(mean=0, SD=1)
// student(t, df) returns
//   integral probability of both side area (< -t) and (t <) of t-dist(df)
const Ps = [-0.4, -3.3, -24, -85.5];
function student(t, df) {
    const a = df - 0.5;
    const b = 48 * a * a;
    const y0 = (t * t) / df;
    const y = a * (y0 > 1e-6 ? Math.log(y0 + 1) : y0);
    const s = (pol(y, Ps) / (0.8 * y * y + 100 + b) + y + 3) / b + 1;
    return 2 * gauss(-s * (y ** 0.5));
}

// welch's t-test
export function tTest(x, y) {
    console.assert(x.length > 1 && y.length > 1);
    const nX = x.length, nY = y.length, nX1 = nX - 1, nY1 = nY - 1;
    const meanX = x.reduce((r, v) => r + v, 0) / nX;
    const meanY = y.reduce((r, v) => r + v, 0) / nY;
    const varX = x.reduce((r, v) => r + (v - meanX) ** 2, 0) / nX1;
    const varY = y.reduce((r, v) => r + (v - meanY) ** 2, 0) / nY1;
    // see: t and nu of https://en.wikipedia.org/wiki/Welch%27s_t-test
    const avX = varX / nX, avY = varY / nY;
    const t = (meanX - meanY) / ((avX + avY) ** 0.5);
    const df = ((avX + avY) ** 2) / ((avX ** 2) / nX1 + (avY ** 2) / nY1);
    const p = student(t, df);
    return {p, t, df, meanX, meanY};
}

// student t-test
export function tTest0(x, y) {
    console.assert(x.length === y.length && x.length > 1);
    const n = x.length, n1 = n - 1;
    const meanX = x.reduce((r, v) => r + v, 0) / n;
    const meanY = y.reduce((r, v) => r + v, 0) / n;
    const varX = x.reduce((r, v) => r + (v - meanX) ** 2, 0) / n1;
    const varY = y.reduce((r, v) => r + (v - meanY) ** 2, 0) / n1;
    // see: https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_sample_sizes,_equal_variance
    const t = (meanX - meanY) / (((varX + varY) / n) ** 0.5);
    const df = n1 * 2; // as 2n-2
    const p = student(t, df);
    return {p, t, df, meanX, meanY};
}

// paired t-test
export function tPairedTest(x, y) {
    console.assert(x.length === y.length && x.length > 1);
    const n = x.length, n1 = n - 1;
    const d = x.map((v, i) => v - y[i]);
    const meanD = d.reduce((r, v) => r + v, 0) / n;
    const varD = d.reduce((r, v) => r + (v - meanD) ** 2, 0) / n1;
    const t = meanD / ((varD / n) ** 0.5);
    const df = n1;
    const p = student(t, df);
    return {p, t, df, meanD, varD};
}
