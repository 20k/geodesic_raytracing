///krasnikov tubes: https://core.ac.uk/download/pdf/25208925.pdf
function krasnikov_thetae(v, e)
{
    return 0.5 * (CMath.tanh(2 * ((2 * v / e) - 1)) + 1);
}

///values here are picked for numerical stability, in particular D should be < the precision bounding box, and its more numerically stable the higher e is
function krasnikov_tube_metric_cart(t, x, y, z)
{
    var e = 0.75; ///width o the tunnel
    var D = 5; ///length of the tube
    var pmax = 2; ///size of the mouth

    ///[0, 2], approx= 0?
    var little_d = 0.01; ///unsure, <1 required for superluminosity

    var p = CMath.sqrt(y * y + z * z);

    var k_t_x_p = 1 - (2 - little_d) * krasnikov_thetae(pmax - p, e) * krasnikov_thetae(t - x - p, e) * (krasnikov_thetae(x, e) - krasnikov_thetae(x + e - D, e));

    var dxdt = 1 - k_t_x_p;

    var ret = [];
    ret.length = 16;

    ret[0 * 4 + 0] = -1;
    ret[1 * 4 + 1] = k_t_x_p;
    ret[2 * 4 + 2] = 1;
    ret[3 * 4 + 3] = 1;
    ret[0 * 4 + 1] = 0.5 * dxdt;
    ret[1 * 4 + 0] = 0.5 * dxdt;

    return ret;
}

krasnikov_tube_metric_cart
