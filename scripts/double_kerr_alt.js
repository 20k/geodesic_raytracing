///https://arxiv.org/pdf/1702.02209.pdf
function double_kerr_alt(t, p, phi, z)
{
    var i = CMath.i;

    CMath.debug(i);

    var R = 4;
    var M = 0.3;
    var q = 0.2;

    var sigma = CMath.sqrt(M*M - q*q * (1 - (4 * M * M * (R * R - 4 * M * M + 4 * q * q)) / CMath.pow(R * (R + 2 * M) + 4 * q * q, 2)));

    var r1 = CMath.sqrt(p*p + CMath.pow(z - R/2 - sigma, 2));
    var r2 = CMath.sqrt(p*p + CMath.pow(z - R/2 + sigma, 2));

    var r3 = CMath.sqrt(p*p + CMath.pow(z + R/2 - sigma, 2));
    var r4 = CMath.sqrt(p*p + CMath.pow(z + R/2 + sigma, 2));

    var littled = 2 * M * q * (R * R - 4 * M * M + 4 * q * q) / (R * (R + 2 * M) + 4 * q * q);

    var pp = 2 * (M*M - q*q) - (R + 2 * M) * sigma + M * R + i * (q * (R - 2 * sigma) + littled);
    var pn = 2 * (M*M - q*q) - (R - 2 * M) * sigma - M * R + i * (q * (R - 2 * sigma) - littled);

    var sp = 2 * (M*M - q*q) + (R - 2 * M) * sigma - M * R + i * (q * (R + 2 * sigma) - littled);
    var sn = 2 * (M*M - q*q) + (R + 2 * M) * sigma + M * R + i * (q * (R + 2 * sigma) + littled);

    var k0 = (R * R - 4 * sigma * sigma) * ((R * R - 4 * M * M) * (M * M - sigma * sigma) + 4 * q * q * q * q + 4 * M * q * littled);
    var kp = R + 2 * (sigma + 2 * i * q);
    var kn = R - 2 * (sigma + 2 * i * q);

    var delta = 4 * sigma * sigma * (pp * pn * sp * sn * r1 * r2 + CMath.conjugate(pp) * CMath.conjugate(pn) * CMath.conjugate(sp) * CMath.conjugate(sn) * r3 * r4)
                        -R * R * (CMath.conjugate(pp) * CMath.conjugate(pn) * sp * sn * r1 * r3 + pp * pn * CMath.conjugate(sp) * CMath.conjugate(sn) * r2 * r4)
                        +(R * R - 4 * sigma * sigma) * (CMath.conjugate(pp) * pn * CMath.conjugate(sp) * sn * r1 * r4 + pp * CMath.conjugate(pn) * sp * CMath.conjugate(sn) * r2 * r3);

    var gamma = -2 * i * sigma * R * ((R - 2 * sigma) * CMath.Imaginary(pp * CMath.conjugate(pn)) * (sp * sn * r1 - CMath.conjugate(sp) * CMath.conjugate(sn) * r4) + (R + 2 * sigma) * CMath.Imaginary(sp * CMath.conjugate(sn)) * (pp * pn * r2 - CMath.conjugate(pp) * CMath.conjugate(pn) * r3));

    var G = 4 * sigma * sigma * ((R - 2 * i * q) * pp * pn * sp * sn * r1 * r2 - (R + 2 * i * q) * CMath.conjugate(pp) * CMath.conjugate(pn) * CMath.conjugate(sp) * CMath.conjugate(sn) * r3 * r4)
                    -2 * R * R * ((sigma - i * q) * CMath.conjugate(pp) * CMath.conjugate(pn) * sp * sn * r1 * r3 - (sigma + i * q) * pp * pn * CMath.conjugate(sp) * CMath.conjugate(sn) * r2 * r4)
                    - 2 * i * q * (R * R - 4 * sigma * sigma) * CMath.Real(pp * CMath.conjugate(pn) * sp * CMath.conjugate(sn)) * (r1 * r4 + r2 * r3)
                    - i * sigma * R * ((R - 2 * sigma) * CMath.Imaginary(pp * CMath.conjugate(pn)) * (CMath.conjugate(kp) * sp * sn * r1 + kp * CMath.conjugate(sp) * CMath.conjugate(sn) * r4)
                                       + (R + 2 * sigma) * CMath.Imaginary(sp * CMath.conjugate(sn)) * (kn * pp * pn * r2 + CMath.conjugate(kn) * CMath.conjugate(pp) * CMath.conjugate(pn) * r3));

    var w = 2 * CMath.Imaginary((delta - gamma) * (z * CMath.conjugate(gamma) + CMath.conjugate(G))) / (CMath.self_conjugate_multiply(delta) - CMath.self_conjugate_multiply(gamma));

    var e2y = (CMath.self_conjugate_multiply(delta) - CMath.self_conjugate_multiply(gamma)) / (256 * sigma * sigma * sigma * sigma * R * R * R * R * k0 * k0 * r1 * r2 * r3 * r4);
    var f = (CMath.self_conjugate_multiply(delta) - CMath.self_conjugate_multiply(gamma)) / CMath.Real((delta - gamma) * (CMath.conjugate(delta) - CMath.conjugate(gamma)));

    var dp = (e2y / f);
    var dz = e2y / f;
    var dphi_1 = p * p / f;
    var dt = -f;
    var dphi_2 = -f * w * w;
    var dt_dphi = f * 2 * w;

    var ret = [];
    ret.length = 16;

    ret[0 * 4 + 0] = dt;
    ret[1 * 4 + 1] = dp;
    ret[2 * 4 + 2] = dphi_1 + dphi_2;
    ret[3 * 4 + 3] = dz;

    ret[0 * 4 + 2] = dt_dphi * 0.5;
    ret[2 * 4 + 0] = dt_dphi * 0.5;

    return ret;
}

double_kerr_alt
