///https://arxiv.org/pdf/1408.6041.pdf is where this formulation comes from
function janis_newman_winicour(t, r, theta, phi)
{
	$cfg.r0.$default = 1;
	$cfg.mu.$default = 4;
	
    var r0 = $cfg.r0;
    ///mu = [1, +inf]
    var mu = $cfg.mu;

    var Ar = CMath.pow((2 * r - r0 * (mu - 1)) / (2 * r + r0 * (mu + 1)), 1/mu);
    var Br = (1/4.) * CMath.pow(2 * r + r0 * (mu + 1), (1/mu) + 1) / CMath.pow(2 * r - r0 * (mu - 1), (1/mu) - 1);

    var dt = -Ar;
    var dr = 1/Ar;
    var dtheta = Br;
    var dphi = Br * CMath.sin(theta) * CMath.sin(theta);

    return [dt, dr, dtheta, dphi];

    ///this formulation has coordinate singularities coming out of its butt
    /*dual q = sqrt(3) * 1.1;
    dual M = 1;
    dual b = 2 * sqrt(M * M + q * q);

    dual gamma = 2*M/b;

    dual dt = -pow(1 - b/r, gamma);
    dual dr = pow(1 - b/r, -gamma);
    dual dtheta = pow(1 - b/r, 1-gamma) * r * r;
    dual dphi = pow(1 - b/r, 1-gamma) * r * r * sin(theta) * sin(theta);

    return {dt, dr, dtheta, dphi};*/
}

janis_newman_winicour
