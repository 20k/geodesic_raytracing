function double_kerr(t, p, phi, z)
{
    var i = CMath.i;

	$cfg.R.$default = 3;
	$cfg.M.$default = 0.3;
	$cfg.a.$default = 0.27;

    ///distance between black holes
    var R = $cfg.R;

    var M = $cfg.M;
    var a = $cfg.a;

    var d = 2 * M * a * (R * R - 4 * M * M + 4 * a * a) / (R * R + 2 * M * R + 4 * a * a);

	$pin(d);

    var sigma_sq = M * M - a * a + (4 * M * M * a * a * (R * R - 4 * M * M + 4 * a * a)) / CMath.pow(R * R + 2 * M * R + 4 * a * a, 2);

	$pin(sigma_sq);

    var sigmap = CMath.sqrt(sigma_sq);
    var sigman = -sigmap;

    var ia = i * a;
    var id = i * d;

    var Rp = ((-M * (2 * sigmap + R) + id) / (2 * M * M + (R + 2 * ia) * (sigmap + ia))) * CMath.sqrt(p * p + CMath.pow((z + 0.5 * R + sigmap), 2));
    var Rn = ((-M * (2 * sigman + R) + id) / (2 * M * M + (R + 2 * ia) * (sigman + ia))) * CMath.sqrt(p * p + CMath.pow((z + 0.5 * R + sigman), 2));

	$pin(Rp);
	$pin(Rn);

    var rp = ((-M * (2 * sigmap - R) + id) / (2 * M * M - (R - 2 * ia) * (sigmap + ia))) * CMath.sqrt(p * p + CMath.pow((z - 0.5 * R + sigmap), 2));
    var rn = ((-M * (2 * sigman - R) + id) / (2 * M * M - (R - 2 * ia) * (sigman + ia))) * CMath.sqrt(p * p + CMath.pow((z - 0.5 * R + sigman), 2));

	$pin(rp);
	$pin(rn);

    //dual K0 = (4 * R * R * sigmap * sigmap * (R * R - 4 * sigmap * sigmap) * ((R * R + 4 * a * a) * (sigmap * sigmap + a * a) - 4 * M * (M * M * M + a * d))) / ((M * M * pow(R + 2 * sigmap, 2) + d * d) * (M * M * pow(R - 2 * sigmap, 2) + d * d));

    var K0 = 4 * sigma_sq * (CMath.pow(R * R + 2 * M * R + 4 * a * a, 2) - 16 * M * M * a * a) / (M * M * (CMath.pow(R + 2 * M, 2) + 4 * a * a));

	$pin(K0);

    var A = R * R * (Rp - Rn) * (rp - rn) - 4 * sigma_sq * (Rp - rp) * (Rn - rn);
    var B = 2 * R * sigmap * ((R + 2 * sigmap) * (Rn - rp) - (R - 2 * sigmap) * (Rp - rn));

	$pin(A);
	$pin(B);

    var G = -z * B + R * sigmap * (2 * R * (Rn * rn - Rp * rp) + 4 * sigmap * (Rp * Rn - rp * rn) - (R * R - 4 * sigma_sq) * (Rp - Rn - rp + rn));

    var w = 4 * a - (2 * CMath.Imaginary(G * (CMath.conjugate(A) + CMath.conjugate(B))) / (CMath.self_conjugate_multiply(A) - CMath.self_conjugate_multiply(B)));

    ///the denominator only has real components
    var f = (CMath.self_conjugate_multiply(A) - CMath.self_conjugate_multiply(B)) / CMath.Real((A + B) * (CMath.conjugate(A) + CMath.conjugate(B)));
    var i_f = CMath.Real((A + B) * (CMath.conjugate(A) + CMath.conjugate(B))) / (CMath.self_conjugate_multiply(A) - CMath.self_conjugate_multiply(B));

    var i_f_e2g = CMath.Real((A + B) * (CMath.conjugate(A) + CMath.conjugate(B))) / CMath.Real(K0 * K0 * Rp * Rn * rp * rn);

    //dual i_f = 1/f;

    ///I'm not sure if the denominator is real... but I guess it must be?
    var e2g = (CMath.self_conjugate_multiply(A) - CMath.self_conjugate_multiply(B)) / CMath.Real(K0 * K0 * Rp * Rn * rp * rn);

    //dual i_f_e2g = i_f * e2g;

    var dphi2 = w * w * -f;
    var dphi1 = i_f * p * p;

    var dt_dphi = f * w * 2;

    var dp = i_f_e2g;
    var dz = i_f_e2g;

    var ret = [];
    ret.length = 16;

    ret[0 * 4 + 0] = -f;
    ret[2 * 4 + 2] = dphi1 + dphi2;
    ret[0 * 4 + 2] = dt_dphi * 0.5;
    ret[2 * 4 + 0] = dt_dphi * 0.5;

    ret[1 * 4 + 1] = dp;
    ret[3 * 4 + 3] = dz;

    return ret;
}

double_kerr
