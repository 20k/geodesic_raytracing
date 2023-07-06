function func(t, r, theta, phi)
{
	function positive_fmod(a, b)
	{	
		var r = CMath.smooth_fmod(a, b);

		var rlt = CMath.lt(r, 0);

		return CMath.select(rlt, r + b, r);
	}
        
	var ret = [t, r, theta, phi];
	
	ret[2] = positive_fmod(ret[2], Math.PI);
	ret[3] = positive_fmod(ret[3], 2*Math.PI);
	
	return ret;
}

func