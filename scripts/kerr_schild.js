
///https://arxiv.org/pdf/0706.0622.pdf
function kerr_schild_metric(t, x, y, z)
{
	$cfg.a.$default = -0.5;
	$cfg.rs.$default = 1;

    var a = $cfg.a;
	var rs = $cfg.rs;

    var R2 = x * x + y * y + z * z;
    var Rm2 = x * x + y * y - z * z;
	
	$pin(R2);
	$pin(Rm2);

    //dual r2 = (R2 - a*a + sqrt((R2 - a*a) * (R2 - a*a) + 4 * a*a * z*z))/2;

    var r2 = (-a*a + CMath.sqrt(a*a*a*a - 2*a*a * Rm2 + R2*R2) + R2) / 2;

	$pin(r2);

    var r = CMath.sqrt(r2);

    var minkowski = [-1, 0, 0, 0,
                      0, 1, 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1];

    var lv = [1, (r*x + a*y) / (r2 + a*a), (r*y - a*x) / (r2 + a*a), z/r];

	$pin(lv);

    var f = rs * r2 * r / (r2 * r2 + a*a * z*z);

	$pin(f);

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
