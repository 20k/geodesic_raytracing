function polar_to_ingoing_ef(t, r, theta, phi)
{
	var rs = $cfg.rs;
	
    var r_star = r + rs * CMath.log((r / rs) - 1);
    
	var v = t + r_star;

    return [v, r, theta, phi];
}

polar_to_ingoing_ef
