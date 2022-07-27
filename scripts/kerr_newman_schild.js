///https://en.wikipedia.org/wiki/Kerr%E2%80%93Newman_metric#Kerr%E2%80%93Schild_coordinates
function kerr_schild_metric(t, x, y, z)
{
	$cfg.a.$default = -0.51;
	$cfg.rs.$default = 1;
	$cfg.Q.$default = 0.51;

    var a = $cfg.a;
	var rs = $cfg.rs;
	var Q = $cfg.Q;

    var R2 = x * x + y * y + z * z;
    var Rm2 = x * x + y * y - z * z;

    var r2 = (-a*a + CMath.sqrt(a*a*a*a - 2*a*a * Rm2 + R2*R2) + R2) / 2;

	$pin(r2);

    var r = CMath.sqrt(r2);

    var minkowski = [-1, 0, 0, 0,
                      0, 1, 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1];

    var lv = [1, (r*x + a*y) / (r2 + a*a), (r*y - a*x) / (r2 + a*a), z/r];

	$pin(lv);

    var f = (rs * r - Q * Q) * r * r / (r2 * r2 + a*a * z*z);
    //dual f = rs * r*r*r / (r*r*r*r + a*a * z*z);

    var g = [];
    g.length = 16;

    for(var a=0; a < 4; a++)
    {
        for(var b=0; b < 4; b++)
        {
            g[a * 4 + b] = minkowski[a * 4 + b] + f * lv[a] * lv[b];
        }
    }

    return g;
}

kerr_schild_metric
