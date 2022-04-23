function ernst_metric(t, r, theta, phi)
{
	$cfg.B.$default = 0.0000025;
	$cfg.rs.$default = 1;
	
    var B = $cfg.B;
	var rs = $cfg.rs;

    var lambda = 1 + B * B * r * r * CMath.sin(theta) * CMath.sin(theta);

    var lambda_sq = lambda * lambda;

    var c = 1;
    var dt = -lambda_sq * (1 - rs/r);
    var dr = lambda_sq * 1/(1 - rs/r);
    var dtheta = lambda_sq * r * r;
    var dphi = r * r * CMath.sin(theta) * CMath.sin(theta) / lambda_sq;

    return [dt, dr, dtheta, dphi];
}

ernst_metric
