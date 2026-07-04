// Non-snake-case names (N, openMin, derivRoots, ...) mirror the notation of the
// paper and its reference implementation.
#![allow(non_snake_case)]

/// Cem Yuksel's polynomial root finder, as described by
/// ``High-Performance Polynomial Root Finding for Graphics'' in
/// Proc. ACM Comput. Graph. Interact. Tech. (Proceedings of HPG 2022)
/// Rust port of the methods in https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h
///
/// Provided under the following license:
///
/// MIT License
///
/// Copyright (c) 2016, Cem Yuksel <cem@cemyuksel.com>
/// All rights reserved.
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in all
/// copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/// SOFTWARE.
use crate::polynomial::Polynomial;
use crate::real::Real;

#[inline]
#[allow(clippy::too_many_arguments)]
fn find_closed<T: Real>(
    N: usize,
    p: impl Fn(T) -> T,
    dp: impl Fn(T) -> T,
    x0: T,
    x1: T,
    y0: T,
    _y1: T,
    tol: T,
) -> T {
    let two = T::from_u32(2);
    let ep2 = two * tol;
    let mut xr = (x0 + x1) / two;
    if x1 - x0 <= ep2 {
        return xr;
    }

    if N <= 3 {
        let xr0 = xr;
        for _ in 0..16 {
            let mut xn = xr - p(xr) / dp(xr);
            // clamp to [x0, x1]
            xn = if xn < x0 {
                x0
            } else if xn > x1 {
                x1
            } else {
                xn
            };
            if (xr - xn).abs() <= tol {
                return xn;
            }
            xr = xn;
        }
        if !xr.is_finite() {
            xr = xr0
        }
    }

    let mut yr = p(xr);
    let mut xb0 = x0;
    let mut xb1 = x1;

    loop {
        let side = (y0 < T::ZERO) != (yr < T::ZERO);
        if side {
            xb1 = xr;
        } else {
            xb0 = xr;
        }
        let dy = dp(xr);
        let dx = yr / dy;
        let xn = xr - dx;
        if (xn > xb0) && (xn < xb1) {
            let stepsize = (xr - xn).abs();
            xr = xn;
            if stepsize > tol {
                yr = p(xr);
            } else {
                let mut xs;
                if tol == T::ZERO {
                    xs = if side {
                        xb1 - T::EPSILON
                    } else {
                        xb0 + T::EPSILON
                    };
                } else {
                    xs = if side { xn - tol } else { xn + tol };
                    if xs == xn {
                        xs = if side {
                            xb1 - T::EPSILON
                        } else {
                            xb0 + T::EPSILON
                        };
                    }
                }
                let ys = p(xs);
                let s = (y0 < T::ZERO) != (ys < T::ZERO);
                if side != s {
                    return xn;
                };
                xr = xs;
                yr = ys;
            }
        } else {
            xr = (xb0 + xb1) / two;
            if xr == xb0 || xr == xb1 || xb1 - xb0 <= ep2 {
                if tol == T::ZERO {
                    let xm = if side { xb0 } else { xb1 };
                    let ym = p(xm);
                    if ym.abs() < yr.abs() {
                        xr = xm;
                    }
                }
                break;
            }
            yr = p(xr);
        }
    }
    xr
}

#[inline]
fn find_open_max<T: Real>(
    N: usize,
    p: impl Fn(T) -> T,
    dp: impl Fn(T) -> T,
    x0: T,
    y0: T,
    tol: T,
) -> T {
    find_open_helper(N, p, dp, x0, y0, x0 + T::ONE, tol, false)
}
#[inline]
fn find_open_min<T: Real>(
    N: usize,
    p: impl Fn(T) -> T,
    dp: impl Fn(T) -> T,
    x1: T,
    y1: T,
    tol: T,
) -> T {
    find_open_helper(N, p, dp, x1, y1, x1 - T::ONE, tol, true)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn find_open_helper<T: Real>(
    N: usize,
    p: impl Fn(T) -> T,
    dp: impl Fn(T) -> T,
    mut xm: T,
    mut ym: T,
    mut xr: T,
    tol: T,
    openMin: bool,
) -> T {
    let mut delta = T::ONE;
    let mut yr = p(xr);

    let mut otherside: bool = (ym < T::ZERO) != (yr < T::ZERO);

    'main_loop: while yr != T::ZERO {
        if otherside {
            if openMin {
                return find_closed(N, p, dp, xr, xm, yr, ym, tol);
            } else {
                return find_closed(N, p, dp, xm, xr, ym, yr, tol);
            }
        } else {
            'open_interval: loop {
                xm = xr;
                ym = yr;
                let dy = dp(xr);
                let dx = yr / dy;
                let xn = xr - dx;
                let dif = if openMin { xr - xn } else { xn - xr }; // Consider using |xr-xn|...
                if dif <= T::ZERO && xn.is_finite() {
                    xr = xn;
                    if dif <= tol {
                        if xr == xm {
                            break 'main_loop;
                        };
                        let xs = if openMin { xn + tol } else { xn - tol };
                        let ys = p(xs);
                        let s = (ym < T::ZERO) != (ys < T::ZERO);
                        if s {
                            break 'main_loop;
                        };
                        xr = xs;
                        yr = ys;
                        continue 'open_interval;
                    }
                } else {
                    xr = if openMin { xr - delta } else { xr + delta };
                    delta += T::from_u32(2);
                }
                yr = p(xr);
                otherside = (ym < T::ZERO) != (yr < T::ZERO);
                continue 'main_loop;
            }
        }
    }

    xr
}

/// Real roots of the quadratic `p.c[0]·x² + p.c[1]·x + p.c[2]`, ascending;
/// slots without a real root are `NAN`.
#[inline]
pub fn roots_quadratic<T: Real>(p: &Polynomial<T, 3>) -> [T; 2] {
    let mut output = [T::NAN; 2];
    let a = p.c[0];
    let b = p.c[1];
    let c = p.c[2];
    let delta = b * b - T::from_u32(4) * a * c;
    if delta > T::ZERO {
        // Two real roots
        let d = delta.sqrt();
        let q = -(b + d.copysign(b)) / T::from_u32(2);
        let rv0 = q / a;
        let rv1 = c / q;
        output[0] = if rv0 < rv1 { rv0 } else { rv1 };
        output[1] = if rv0 < rv1 { rv1 } else { rv0 };
        return output;
    } else if delta < T::ZERO { // Roots are complex conjugate pair, return NaNs
    } else {
        // One real root
        output[0] = -b / (T::from_u32(2) * a);
    }

    output
}

/// Real roots of the cubic `f.c[0]·x³ + … + f.c[3]`, ascending; slots without
/// a real root are `NAN`. Finite roots always form a prefix of the output.
#[inline]
pub fn roots_cubic<T: Real>(f: &Polynomial<T, 4>, tol: T) -> [T; 3] {
    let mut output = [T::NAN; 3];
    let a = f.c[0] * T::from_u32(3);
    let b_2 = f.c[1];
    let c = f.c[2];

    let df = Polynomial::<T, 3>::new([a, T::from_u32(2) * b_2, c]);
    let p = { |x| f.eval(x) };
    let dp = { |x| df.eval(x) };

    let delta_4 = b_2 * b_2 - a * c;

    if delta_4 > T::ZERO {
        let d_2 = delta_4.sqrt();
        let q = -(b_2 + d_2.copysign(b_2));
        let rv0 = q / a;
        let rv1 = c / q;
        let xa = if rv0 < rv1 { rv0 } else { rv1 };
        let xb = if rv0 < rv1 { rv1 } else { rv0 };

        let ya = p(xa);
        let yb = p(xb);

        if (a < T::ZERO) == (ya < T::ZERO) {
            output[0] = find_open_min(3, p, dp, xa, ya, tol);
            if (ya < T::ZERO) != (yb < T::ZERO) {
                output[1] = find_closed(3, p, dp, xa, xb, T::ZERO, T::ZERO, tol);
                output[2] = find_open_max(3, p, dp, xb, yb, tol);
            }
        } else {
            output[0] = find_open_max(3, p, dp, xb, yb, tol);
        }
    } else {
        let x_inf = -b_2 / a;
        let y_inf = p(x_inf);
        if (a < T::ZERO) != (y_inf < T::ZERO) {
            output[0] = find_open_max(3, p, dp, x_inf, y_inf, tol);
        } else {
            output[0] = find_open_min(3, p, dp, x_inf, y_inf, tol);
        }
    }

    output
}

/// Real roots of the quartic `f.c[0]·x⁴ + … + f.c[4]`, ascending; slots
/// without a real root are `NAN`. Finite roots always form a prefix of the
/// output.
#[inline]
pub fn roots_quartic<T: Real>(f: &Polynomial<T, 5>, tol: T) -> [T; 4] {
    const N: usize = 4;
    let mut output = [T::NAN; N];
    let df = Polynomial::<T, 4>::new([
        T::from_u32(4) * f.c[0],
        T::from_u32(3) * f.c[1],
        T::from_u32(2) * f.c[2],
        f.c[3],
    ]);
    let derivRoots = roots_cubic(&df, tol);

    let p = { |x| f.eval(x) };
    let dp = { |x| df.eval(x) };
    let nd = derivRoots
        .iter()
        .map(|x| if x.is_finite() { 1 } else { 0 })
        .sum::<usize>();
    if ((N & 1) != 0) || ((N & 1) == 0 && nd > 0) {
        let mut nr = 0;
        let mut xa = derivRoots[0];
        let mut ya = p(xa);
        if ((ya < T::ZERO) != (f.c[0] < T::ZERO)) != ((N & 1) != 0) {
            output[0] = find_open_min(N, p, dp, xa, ya, tol);
            nr = 1;
        }
        for i in 1..nd {
            let xb = derivRoots[i];
            let yb = p(xb);
            if (ya < T::ZERO) != (yb < T::ZERO) {
                output[nr] = find_closed(N, p, dp, xa, xb, ya, yb, tol);
                nr += 1;
            }
            xa = xb;
            ya = yb;
        }
        if (ya < T::ZERO) != (f.c[0] < T::ZERO) {
            output[nr] = find_open_max(N, p, dp, xa, ya, tol);
            // nr += 1;
        }
    }

    output
}
