function polar_to_cartesian(t, r, theta, phi)
{
    var x = r * CMath.sin(theta) * CMath.cos(phi);
    var y = r * CMath.sin(theta) * CMath.sin(phi);
    var z = r * CMath.cos(theta);

    return [x, t, y, z];
}

polar_to_cartesian
