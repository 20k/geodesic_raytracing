///krasnikov tubes: https://core.ac.uk/download/pdf/25208925.pdf
function krasnikov_thetae(v, e)
{
    return 0.5 * (CMath.tanh(2 * ((2 * v / e) - 1)) + 1);
}

///they call x, z
///krasnikov is extremely terrible, because its situated down the z axis here which is super incredibly bad for performance
function krasnikov_tube_metric(t, p, phi, x)
{
	$cfg.e.$default = 0.1;
	$cfg.D.$default = 2;
	$cfg.pmax.$default = 1;
	
    var e = 0.1; ///width o the tunnel
    var D = 2; ///length of the tube
    var pmax = 1; ///size of the mouth

    ///[0, 2], approx= 0?
    var little_d = 0.01; ///unsure, <1 required for superluminosity

    var k_t_x_p = 1 - (2 - little_d) * krasnikov_thetae(pmax - p, e) * krasnikov_thetae(t - x - p, e) * (krasnikov_thetae(x, e) - krasnikov_thetae(x + e - D, e));

    var dxdt = 1 - k_t_x_p;

    var ret = [];
    ret.length = 16;

    ret[0 * 4 + 0] = -1;
    ret[1 * 4 + 1] = 1;
    ret[2 * 4 + 2] = p * p;
    ret[3 * 4 + 3] = k_t_x_p;
    ret[0 * 4 + 3] = 0.5 * dxdt;
    ret[3 * 4 + 0] = 0.5 * dxdt;

    return ret;
}

krasnikov_tube_metric
